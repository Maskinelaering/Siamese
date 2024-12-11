import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import gc

from torch.utils.tensorboard import SummaryWriter
import tqdm as tqdm
from sklearn import metrics
from sklearn.manifold import TSNE
import ray
from ray.tune.schedulers import ASHAScheduler
from functools import partial

import data_setup, utils, tuning

#model_interpreter.run = False

"""
This script is for all the main running parts:
    training, incl. contrastive loss function and early stopping
    create_writer, for saving the training details for comparison
    testing, incl. saving output embeddings and predictions

"""



"########################################## TRAINING ###########################################"
##### ------------- Step 3: Define training function ------------

# Define the Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function used to push similar images to a lower distance D
    and dissimilar images to a higher distance. 

    Inputs:
        prediction: cosine_similarity of the two input images, i.e. the distance between the 
                    embedded outputs (final fully-connected layer)
                    
        
        truth:      cosine_similarity of the metadata of the two input images, i.e. the true 
                    value of similarity.
                    Transformed in this function to the target value from [0,1] where
                    truth=1 -> target=1, truth=-1 -> target=0

        data_setup.distance_function:
                    if == "cosine": 
                        Transforms prediction to a distance value from [-1,1]->[0,1] 
                        with distance = 0.5 * (1 - prediction),
                        such that prediction=1 -> distance=0 and prediction=(-1) -> distance=1
                        &
                        Transforms truth to a target value from [-1,1]->[0,1] 
                        with target = 0.5 * (truth + 1),
                        such that prediction=1 -> distance=0 and prediction=(-1) -> distance=1

        Drives similar images (truth~1, target~1) towards a smaller distance (higher predictions)
        and dissimilar images (truth~-1, target~0) towards a larger distance (lower predictions),
        but maximum distance=1 (pred=-1)

    NOTE: The outputs are unused here since it just uses the predictions 
    from the model and the target true value.
    """
    def __init__(self, margin: float):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, prediction, truth):
        
        if data_setup.distance_function == "cosine":
            distance = 0.5 * (1 - prediction) # pred=1 -> dist=0, pred=-1 -> dist=1

            target = 0.5 * (truth + 1) # truth=1 -> target=1, truth=-1 -> target=0
            
            first_part = 0.5 * (target) * torch.pow(distance, 2)
            second_part = 0.5 * (1 - target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

            loss_contrastive = first_part + second_part

        if data_setup.distance_function == "euclid":
            distance = prediction
            loss_contrastive = torch.mean((1 - target) * torch.pow(distance, 2) +
                                     (target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        
        return loss_contrastive, first_part, second_part


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        if val_loss < 0:
            self.early_stop = True



def training(model: nn.Module,
             train_dataloader: DataLoader,
             validation_dataloader: DataLoader,
             output_dir: str, 
             model_name: str,
             loss_fn: nn.Module,
             optimizer: optim.Optimizer,
             num_epochs: int,
             device: torch.device,
             writer: SummaryWriter,
             patience: int,
             min_delta: float):
    
    print("[INFO] Training started")

    model_eval_metric = "val_loss"
    
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    scaler = torch.cuda.amp.GradScaler() # https://pytorch.org/docs/stable/amp.html#gradient-scaling

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.1,
                                                            patience=5,
                                                            verbose=True
                                                                )
    # ------TRAINING------
    training_losses = []
    first_train_losses = []
    second_train_losses = []
    train_mses = []

    validation_losses = []
    first_val_losses = []
    second_val_losses = []
    val_mses = []
    epochs = []

    best_validation_loss = 1e9

    try:
        model_parameter_df = pd.DataFrame(columns=[name for name, _ in model.named_parameters() if _.grad is not None])
    except Exception as e:
        print("There was an error initializing the gradient dataframe. Returning None.")
        model_parameter_df = None
        print("Error:", e)

    for epoch in range(num_epochs):
        model.train()
        training_loss_output = []
        train_first_output = []
        train_second_output = []
        train_truths = []
        train_predictions = []

        for i, batch in enumerate(train_dataloader):
            #print("bshape", batch)
            img1, img2 = batch[0], batch[1] # OBS: Model takes as input image sets when working with batches
            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            if batch[2].dtype != torch.float:
                sim_score = batch[2].to(device, dtype=torch.float)
            else:
                sim_score = batch[2].to(device).float()
            if np.isnan(sim_score.detach().cpu().numpy()).any():
                print("sim score index with nan", i)
            

            optimizer.zero_grad() # zero the gradients so they dont accumulate over epochs
            with torch.autocast(device_type="cuda", dtype=torch.float16): 
                # run in mixed precision to improve performance while maintaining accuracy
                # See more: https://pytorch.org/docs/stable/amp.html#autocasting

                # Forward pass
                output1, output2, prediction = model(img1, img2)

                if np.isnan(prediction.detach().cpu().numpy()).any():
                    print("There was a NaN value in the batch predictions at index", i)
                    print(prediction)
                    print("Setting NaN-value to zero")
                    prediction = torch.nan_to_num(prediction, nan=0.1)
                    print(prediction)
                assert output1.dtype is torch.float16
                assert output2.dtype is torch.float16

                # Calculate loss
                training_loss, train_first, train_second = loss_fn(output1, output2, 
                                                       prediction, sim_score) 
                assert training_loss.dtype is torch.float32
                assert train_first.dtype is torch.float32
                assert train_second.dtype is torch.float32

                output1.detach() # No need to calculate gradients of these items
                output2.detach()
                train_first.detach()
                train_second.detach()

                #torch.cuda.empty_cache() # empty after each batch?
                

            # Perform backpropagation
            scaler.scale(torch.mean(training_loss)).backward(retain_graph=True)

            # Take a gradient descent step by the optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Save values 
            training_loss_output.append(torch.mean(training_loss).item())
            train_first_output.append(torch.mean(train_first).item())
            train_second_output.append(torch.mean(train_second).item())
            train_truths.append(sim_score.detach().cpu().numpy())
            train_predictions.append(prediction.detach().cpu().numpy())


        training_loss = np.mean(training_loss_output) # Getting the mean loss for the batch
        first_training_loss = np.mean(train_first_output)
        second_training_loss = np.mean(train_second_output)

        try:
            train_mse = metrics.mean_squared_error(np.concatenate(train_truths), np.concatenate(train_predictions))
            train_mses.append(train_mse)
        except Exception as e:
            print(f"There was an error in MSE calculation: {e}")
            
            train_mses.append(1.0)

        scheduler.step(training_loss) # Updates learning rate

        training_losses.append(training_loss)
        first_train_losses.append(first_training_loss)
        second_train_losses.append(second_training_loss)
        

        if writer:
            # Add results to SummaryWriter
            writer.add_scalar(tag="Training loss", 
                            scalar_value=training_loss,
                            global_step=epoch
                            )
            
            writer.add_graph(model=model, 
                             input_to_model = (img1, img2)
                            )
            writer.close()

        #torch.cuda.empty_cache() # empty before evaluation?
        #gc.collect()

        model.eval()  # Set the model to evaluation mode
        validation_loss_output = []
        val_first_output = []
        val_second_output = []
        val_truths = []
        val_predictions = []

        for batch in validation_dataloader:
            val_img1, val_img2 = batch[0], batch[1]
            val_img1 = val_img1.to(device).float()
            val_img2 = val_img2.to(device).float()
            val_sim_score = batch[2].to(device).float()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                val_output1, val_output2, val_prediction = model(val_img1, val_img2) 
                assert val_output1.dtype is torch.float16
                assert val_output2.dtype is torch.float16
                validation_loss, val_first, val_second = loss_fn(val_output1, val_output2, val_prediction, val_sim_score) 
                assert validation_loss.dtype is torch.float32
                
                val_output1.detach()
                val_output2.detach()
                val_first.detach()
                val_second.detach()


                validation_loss_output.append(torch.mean(validation_loss).item())
                val_first_output.append(torch.mean(val_first).item())
                val_second_output.append(torch.mean(val_second).item())
                val_truths.append(val_sim_score.detach().cpu().numpy())
                val_predictions.append(val_prediction.detach().cpu().numpy())
        

        validation_loss = np.mean(validation_loss_output)
        first_val_loss = np.mean(val_first_output)
        second_val_loss = np.mean(val_second_output)
        
        try:
            val_mse = metrics.mean_squared_error(np.concatenate(val_truths), np.concatenate(val_predictions))
            val_mses.append(val_mse)
        except Exception as e:
            
            print(f"There was an error in validation MSE calculation: {e}")
            print("Val truths:", np.concatenate(val_truths))
            print("NaN at index:", np.where(np.isnan(val_truths)))
            print("Val predictions", np.concatenate(val_predictions))
            print("NaN at index:", np.where(np.isnan(val_predictions)))
            val_mses.append(1.0)
        validation_losses.append(validation_loss)
        first_val_losses.append(first_val_loss)
        second_val_losses.append(second_val_loss)
        
        # Save improved model, if metric has improved
        if model_eval_metric == "val_loss":
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                utils.save_model(model, output_dir, model_name, structure_df=None)

        epochs.append(epoch+1)

        print(f'Epoch:{epoch+1}, Training Loss: {training_loss:.6f}, Validation Loss: {validation_loss:.6f}')
        
        epoch_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                epoch_gradients[name] = param.grad.norm().item()
        try:
            model_parameter_df = pd.concat([model_parameter_df, pd.DataFrame(epoch_gradients, index=[0])], ignore_index=True)
        except:
            print("There was an error in concatenating the gradient dataframe. Returning None")
            model_parameter_df = None
            print("Error:", e)

        h5_filename = os.path.join(output_dir, model_name, "training_stats.h5")
        os.makedirs(os.path.dirname(h5_filename), exist_ok=True)
        with h5py.File(h5_filename, "w") as hf:
            batch_id = "all_batches"
            train_val_params = {"epoch": epochs, 
                            "training_loss": training_losses, 
                            "first_train_loss": first_train_losses,
                            "second_train_loss": second_train_losses,
                            "train_mse": train_mses,
                            "validation_loss": validation_losses,
                            "first_val_loss": first_val_losses,
                            "second_val_loss": second_val_losses,
                            "validation_mse": val_mses,
                            }
            utils.save_batch_data_hdf5(hf, train_val_params, batch_id)
        if writer:
            writer.add_scalar(tag="Validation Loss",
                            scalar_value=np.mean(validation_loss_output),
                            global_step=epoch)
        
        early_stopping(validation_loss)

        if writer:
            writer.close()

        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break 
        
        # Calculate memory usage after each epoch
        current_memory = torch.cuda.memory_allocated(0)
        print(f"Total memory usage {current_memory / (1024**2):.2f} MB")

        torch.cuda.empty_cache()
        #gc.collect()

    print("[INFO] Training done")
    return model_parameter_df



def pre_tuning(config, 
           num_samples,
           max_num_epochs,
           gpus_per_trial,
           cpus,
           dataloader_dir,
           output_dir,
           model_name):
    checkpoint_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/checkpoints"

    scheduler = ASHAScheduler(
        metric="val_mse", 
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    print("RUNNING")
    result = ray.tune.run(
        partial(tuning.train_func, data_dir=dataloader_dir),
        resources_per_trial={"cpu": cpus, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir=checkpoint_dir,
    )
    print("RUNNING2")

    top_trials = result.trials
    top_trials.sort(key=lambda trial: trial.last_result.get("val_mse", float("inf")))

    n_saves = len(top_trials)


    # Print the configurations and validation losses of the top three trials
    for i, trial in enumerate(top_trials[:n_saves]):
        print(f"Top {i + 1} trial config: {trial.config}")
        print(f"Top {i + 1} trial final validation loss: {trial.last_result['val_loss']}")
        print(f"Top {i + 1} trial final mean squared error: {trial.last_result['val_mse']}")

        print("\n")


    # Save the output to a text file
    file_path = os.path.join(output_dir, model_name)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Saving as csv file with pandas
    params = [trial.config for trial in top_trials[:n_saves]]
    df = pd.DataFrame(params)
    df['Validation Loss'] = [trial.last_result['val_loss'] for trial in top_trials[:n_saves]]
    df['Mean squared error'] = [trial.last_result['val_mse'] for trial in top_trials[:n_saves]]
    csv_file = os.path.join(file_path, f"tuning_nsamp{num_samples}.csv")
    df.to_csv(csv_file, sep="\t", index=False)

    
    return top_trials




def create_writer(experiment_name: str, 
                  model_name: str,
                  timestamp: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)








"########################################## TESTING ###########################################"


##### --------------- Visualisations -------------------

def visualize_embeddings(h5_filename, output_dir, model_name):
    out1, out2 = utils.get_batch_data_hdf5(h5_filename, ["out1", "out2"])

    tsne = TSNE(n_components=2, random_state=42)
    embed = np.vstack((out1, out2)).T

    embeddings_2d = tsne.fit_transform(embed)
    save_dir = os.path.join(output_dir, model_name, "images")
    save_path = os.path.join(save_dir, "embedding.png")

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    #plt.colorbar()
    plt.title("t-SNE Visualization of Embeddings")
    plt.savefig(save_path)
    plt.close()
    print("Embeddings saved!")


# ------------------- TEST --------------------

def testing(model: nn.Module,
            test_dataloader: DataLoader,
            device: torch.device,
            output_dir: str,
            model_name: str,
):  
    print("\n[INFO] Model test started.")
    print("\nLoading trained model...")
    checkpoint = torch.load(f"{output_dir}/{model_name}/{model_name}.pth")

    model_state_dict = model.state_dict()
    model.eval()
    model.load_state_dict(checkpoint)

    print(f"Testing on {len(test_dataloader)} test batches...")

    # Create an HDF5 file to store images, truths, and predictions
    h5_filename = os.path.join(output_dir, model_name, "batch_data.h5")
    h5_images = os.path.join(output_dir, model_name, "batch_images.h5")
    os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

    with h5py.File(h5_filename, "w") as hf:
        with h5py.File(h5_images, "w") as hf2:
            with torch.no_grad():
                for batch_id, batch in enumerate(test_dataloader):
                    img1_batch, img2_batch = batch[0], batch[1]
                    out1_batch, out2_batch, predictions_batch = model(img1_batch.to(device).float(), 
                                                                img2_batch.to(device).float())
                    truths_batch = batch[2]
                    
                    metadata1 = batch[3]
                    metadata2 = batch[4]
                    metadata1_raw = batch[5]
                    metadata2_raw = batch[6]

                    test_params = {
                            "out1": out1_batch, "out2": out2_batch,
                            "truths": truths_batch, "predictions": predictions_batch,
                            "metadata1": metadata1, "metadata2": metadata2,
                            "metadata1_raw": metadata1_raw, "metadata2_raw": metadata2_raw,
                            }
                    input_images = {
                            "img1": img1_batch, "img2": img2_batch,
                            }
                    utils.save_batch_data_hdf5(hf, test_params, batch_id)
                    utils.save_batch_data_hdf5(hf2, input_images, batch_id)

                    del test_params, input_images
                    #torch.cuda.empty_cache()
                    #gc.collect()
        
    print("[INFO] Test done.")
    # Calculate memory usage after each epoch
    current_memory = torch.cuda.memory_allocated(0)

    print(f"Total memory usage {current_memory / (1024**2):.2f} MB")

    torch.cuda.empty_cache()
    gc.collect()
    # Run functions
    mse = utils.calculate_mse_from_hdf5(h5_filename, )
    print(f"Mean Squared Error: {mse}")
    

    utils.create_clustering_plot(h5_filename, output_dir, model_name, batch_id)

    acc_means, acc_sums = utils.plot_accuracy(h5_filename, output_dir, model_name, lower_limit=0.9, upper_limit=1.0, bins=10, plot=True)
    #visualize_embeddings(h5_filename, output_dir, model_name)

    return 




