from functools import partial
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray.tune.schedulers import ASHAScheduler
from sklearn import metrics

import data_setup, engine, model_builder, utils


"""
This script is for tuning hyperparameters of a given model

"""
"---------------Step 1: Configure the search space and data -----------------"

DATASET_DIR = "/lustre/astro/antonmol/learning_stuff/siamese_networks/datasets_krisha"
TRAIN_SIZE = 0.2

config = {
            "l1": ray.tune.choice([16, 32, 64]), #16, 32
            "l2": ray.tune.choice([32, 64, 128]), #32, 64
            "l3": ray.tune.choice([32, 64, 128]), #32, 64, 128
            "l4": ray.tune.choice([32, 64, 128]), #32, 64, 128
            "l5": ray.tune.choice([64, 128, 256]), #64, 128, 256
            "fc1": ray.tune.choice([512, 2048]), #512, 1024
            "fc2": ray.tune.choice([128]), #64, 128
            "lr": ray.tune.choice([8e-5]), #1e-4, 1e-3, 1e-2
            "batch_size": ray.tune.choice([4]), #64
            "margin": ray.tune.choice([0.8, 1.0]), 
            }

# At each trial, Ray Tune will now randomly sample a combination of parameters from these search spaces. 
# It will then train a number of models in parallel and find the best performing one among these. 
# We also use the ASHAScheduler which will terminate bad performing trials early.

MODEL_NAME = "tuning_SiameseNetwork"

os.environ["CUDA_VISIBLE_DEVICES"]="3" # Choose cuda device
os.environ["RAY_DEDUP_LOGS"] = "0"

transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.CenterCrop(400), # Crop image to 400x400 pixels 
    ])


"--------------Step 2: define training function-----------"
def train_func(config):
    global transform
    #ray.tune.utils.wait_for_gpu()
    model = model_builder.SiameseNetwork(config["l1"], config["l2"],
                    config["l3"], config["l4"],
                    config["l5"], config["fc1"], 
                    config["fc2"])
    
    os.environ["CUDA_VISIBLE_DEVICES"]="3"  # Choose which device to use (astro01 has 4 gpu's)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    loss_fn = engine.ContrastiveLoss(margin=config["margin"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    checkpoint = ray.train.get_checkpoint() #checkpoint = session.get_checkpoint()
    
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_state = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    
    train_dataloader, val_dataloader, _  = data_setup.create_dataloaders(
                                                                    data_setup.data_folder,
                                                                    data_setup.md_names,
                                                                    data_setup.normalizer,
                                                                    data_setup.dataloader_dir,
                                                                    data_type="pkl",
                                                                    train_size=data_setup.TRAIN_SIZE,
                                                                    transform=transform, 
                                                                    batch_size=data_setup.BATCH_SIZE, 
                                                                    num_workers=data_setup.NUM_WORKERS,
                                                                    random_seed=data_setup.RANDOM_SEED,
                                                                    distance_function=data_setup.distance_function
                                                                    )


    scaler = torch.cuda.amp.GradScaler() 

    for epoch in range(start_epoch, 5):  # loop over the dataset multiple times
        model.train()

        for batch in train_dataloader:
            img1, img2 = batch[0], batch[1] # NOTE: Model takes as input image sets when working with batches
            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            sim_score = batch[2].to(device).float()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output1, output2, prediction = model(img1, img2) 
                assert output1.dtype is torch.float16
                assert output2.dtype is torch.float16
                training_loss, _, _ = loss_fn(output1, output2, prediction, sim_score) 
                assert training_loss.dtype is torch.float32

                #training_loss_output.append(training_loss.item())
            scaler.scale(torch.mean(training_loss)).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        
        model.eval()  # Set the model to evaluation mode
        validation_loss_output = []
        truths = []
        predictions = []
        for batch in val_dataloader:
            val_img1, val_img2 = batch[0], batch[1]
            val_img1 = val_img1.to(device).float()
            val_img2 = val_img2.to(device).float()
            val_truth = batch[2].to(device).float()
            truths.append(val_truth.detach().cpu().numpy())

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                val_output1, val_output2, val_prediction = model(val_img1, val_img2) 
                predictions.append(val_prediction.detach().cpu().numpy())
                assert val_output1.dtype is torch.float16
                assert val_output2.dtype is torch.float16
                #assert val_similarity.dtype is torch.float16
                validation_loss, _, _ = loss_fn(val_output1, val_output2, val_prediction, val_truth) 
                assert validation_loss.dtype is torch.float32
                
                validation_loss_output.append(torch.mean(validation_loss).item())
                #predictions.append(val_prediction.detach().cpu().numpy())
                #truths.append(val_sim_score.detach().cpu().numpy())

            scaler.scale(torch.mean(validation_loss)).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        validation_loss = np.mean(validation_loss_output)


        val_mse = metrics.mean_squared_error(np.concatenate(truths), np.concatenate(predictions))
        # Report losses
        ray.train.report({"val_loss": validation_loss, "val_mse": val_mse})

        torch.cuda.empty_cache()





"---------------------Step 3: Tune!------------------------"
def main(config, #dataloader_dir, 
         num_samples=10, max_num_epochs=10, cpus=4, gpus_per_trial=1):
    
    
    scheduler = ASHAScheduler(
        metric="val_mse",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    
    result = ray.tune.run(
        partial(train_func),#, dataloader_dir=dataloader_dir),
        resources_per_trial={"cpu": cpus, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir=checkpoint_dir
    )

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
    tuning_path = "/lustre/astro/antonmol/learning_stuff/siamese_networks/hp_tuning"
    file_path = os.path.join(tuning_path, MODEL_NAME)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    "Saving as csv file with pandas"
    params = [trial.config for trial in top_trials[:n_saves]]
    df = pd.DataFrame(params)
    df['Validation Loss'] = [trial.last_result['val_loss'] for trial in top_trials[:n_saves]]
    df['Mean squared error'] = [trial.last_result['val_mse'] for trial in top_trials[:n_saves]]
    csv_file = os.path.join(file_path, f"nsamp{num_samples}_best_hyperparameters.csv")
    df.to_csv(csv_file, sep="\t", index=False)

    
    return top_trials #result

"------------Step 4: Run tuner ------------"

if __name__ == "__main__":
    
    """
    num_samples = the number of different configurations to run
    gpus_per_trail = fraction of total gpu to use per configuration. 
        Example: gpus_per_trail=0.2 will result in 5 configurations running simultaneously with each 20% of gpu
    cpus = number of cpus to run for each simultaneous iteration
        Example: If 5 configurations run simultaneously, cpus=1 will use 5 cpus.
    max_num_epochs = maximum number of epochs each configuration will run. May run fewer due to early-stopping
    
    """

    num_samples = 400
    max_num_epochs = 10

    gpus_per_trial = 0.25
    cpus = 40 / (1/gpus_per_trial)
    
    result = main(config=config,
                  #dataloader_dir = DATALOADER_DIR,
                  num_samples=num_samples, 
                  max_num_epochs=max_num_epochs, 
                  cpus=cpus, 
                  gpus_per_trial=gpus_per_trial)
    
    

    
