if __name__ == "__main__":
    print("Loading modules...")

import os
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import data_setup, model_builder, engine, utils, matching, tuning
import matplotlib.pyplot as plt
import argparse
import ray
from sklearn import metrics
from ray.tune.schedulers import ASHAScheduler
from functools import partial


""" 
This script is for training a Siamese network to find features of pairs of images
and labeling them on similarities determined by their metadata, such as mass, accretion rate
and age.

The pipeline is as follows:

0. Hyperparameter tuning
    See tuning.py

1. Create or load dataset, including calculating the similarity labels from metadata
    See data_setup.py
2. Create the Siamese model
    See model_builder.py
3. Train and test the model, save output of test
    See engine.py
4. Inspect outputs, assess model accuracy
    See utils.py

"""

# Default values
# TRAIN_SIZE = 0.01
# BATCH_SIZE = 4
# NUM_WORKERS = 1
# NUM_EPOCHS = 10
# MARGIN = 0.9
# PATIENCE = 5
# MIN_DELTA = 1e-3
# LR = 1e-4
# L1 = 32
# L2 = 32
# L3 = 128
# L4 = 64
# L5 = 128
# FC1 = 512
# FC2 = 128
# targets = -1
# RANDOM_SEED = None
# output_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs"
# model_name = "just_testing"
# data_folder = "/lustre/astro/antonmol/atm_new_dataset_sink49"
# dataloader_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders"


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Train a Siamese Network")

parser.add_argument("--train_size", type=float, default=0.01, help="Train size")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--num_workers", type=int, default=1, help="...")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train model over")
parser.add_argument("--margin", type=float, default=1.0, help="Least distance between similar and non-similar image pairs. Criteria for contrastiv loss function")
parser.add_argument("--patience", type=int, default=3, help="For early-stopping: Number of consecutive epochs without improvement to val_loss for early stopping.")
parser.add_argument("--min_delta", type=float, default=1e-4, help="For early-stopping: Minimum improvement in val_loss for early stopping.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--l1", type=int, default=32, help="Number of channels at layers with l1 parameter. Check model for clarification.")
parser.add_argument("--l2", type=int, default=32, help="Number of channels at layers with l2 parameter. Check model for clarification.")
parser.add_argument("--l3", type=int, default=128, help="Number of channels at layers with l3 parameter. Check model for clarification.")
parser.add_argument("--l4", type=int, default=64, help="Number of channels at layers with l4 parameter. Check model for clarification.")
parser.add_argument("--l5", type=int, default=128, help="Number of channels at layers with l5 parameter. Check model for clarification.")
parser.add_argument("--fc1", type=int, default=512, help="Number of nodes in first FC layer. Check model for clarification.")
parser.add_argument("--fc2", type=int, default=128, help="Number of nodes in second FC layer, final output. Check model for clarification.")
parser.add_argument("--targets", type=utils.parse_targets, default=-1, help="The output node from the final FC layer to investigate in model interpretation")
parser.add_argument("--num_samples", type=int, default=20, help="Number of configurations to try for HP tuning.")
parser.add_argument("--max_num_epochs", type=int, default=10, help="The maximum number of epochs pr. configuration for HP tuning.")

parser.add_argument("--output_dir", type=str, default="/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs", help="Path to folder for storing outputs such as trained model, plots and latent feature representations (not dataloaders)")
parser.add_argument("--model_name", type=str, default="just_testing", help="Model name")
parser.add_argument("--data_folder", type=str, default="/lustre/astro/antonmol/atm_new_dataset_sink49", help="Path to datafolder")
parser.add_argument("--dataloader_dir", type=str, default="/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders", help="Path to folder for storing dataloaders")

parser.add_argument("--random_seed", type=int, default=None, help="Random seed for data(?)")

args = parser.parse_args()

########## Learning params ##########
data_setup.BATCH_SIZE = args.batch_size  # Number of inputs in each batch for *multi-core processing*?
data_setup.TRAIN_SIZE = args.train_size # Fraction of data used for training. The remainder is split evenly for validation and testing
data_setup.NUM_WORKERS = args.num_workers # 

data_setup.NUM_EPOCHS = args.num_epochs # Number of iterations. Model sees same input and updates its weights each iteration.
data_setup.MARGIN = args.margin    # Minimum distance
data_setup.LEARNING_RATE = args.lr
data_setup.PATIENCE = args.patience    # For EarlyStopping. Number of consecutive epochs without MIN_DELTA improvement to val_loss before stopping early
data_setup.MIN_DELTA = args.min_delta # For EarlyStopping. Minimal improvement for loss needed to not trigger early stopping.
data_setup.RANDOM_SEED = args.random_seed

data_setup.L1 = args.l1
data_setup.L2 = args.l2
data_setup.L3 = args.l3
data_setup.L4 = args.l4
data_setup.L5 = args.l5
data_setup.FC_UNITS1 = args.fc1
data_setup.FC_UNITS2 = args.fc2


plot_matches = True
device_nr = "1"     
data_type = 'pkl'


########## Directory params ##########

data_setup.output_dir = args.output_dir
data_setup.model_name = args.model_name
data_setup.data_folder = args.data_folder
data_setup.dataloader_dir = args.dataloader_dir

# Create folders for saving output and model
if not os.path.exists(data_setup.output_dir):
    os.mkdir(data_setup.output_dir)


########## Metadata params ##########

#md_names = data_setup.md_names

data_setup.distance_function = "cosine"  # cosine, euclid
data_setup.norm_type = "z_score" # minmax, z_score
#DISTANCE_FUNCTION = data_setup.distance_function  # cosine, euclid

normalizer_file = "metadata_stats_sink49.csv"
normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/{normalizer_file}", index_col=0)
data_setup.normalizer = normalizer

########## Tuning params ##########
num_samples = args.num_samples
max_num_epochs = args.max_num_epochs
gpus_per_trial = 0.25
cpus = 40 / (1/gpus_per_trial)



########## Create/load dataset ##########



data_setup.input_size = 400 # Change input image size if using cropping
print("Input image size:", data_setup.input_size)

# Setup target device
os.environ["CUDA_VISIBLE_DEVICES"] = device_nr  # Choose which device to use (astro01 has 4 gpu's)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available
print(f"\nUsing {device} as device")
print("Current device:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Device name", torch.cuda.get_device_name(torch.cuda.current_device()))


model = model_builder.SiameseNetwork(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2)

# model = model_builder.SiameseNetwork_fcs(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
#                                                 fc_units1=512, fc_units2=1028,
#                                                 fc_units3=4096, fc_units4=128, fc_units5=None)


# Matching params
nsamp = 50
threshold = None
target_sink = "164"
target_projection = "3"
target_normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_sink{target_sink}.csv", index_col=0)

target_snapshots = ["000454"]#, "000578", "000638", "000690"]


structure_dict = {
                    "model_name": data_setup.model_name,
                    "train_size": data_setup.TRAIN_SIZE, 
                    "input_size": data_setup.input_size,
                    "batch_size": data_setup.BATCH_SIZE,
                    "l1": data_setup.L1, 
                    "l2": data_setup.L2, 
                    "l3": data_setup.L3, 
                    "l4": data_setup.L4, 
                    "l5": data_setup.L5, 
                    "fc1": data_setup.FC_UNITS1, 
                    "fc2": data_setup.FC_UNITS2,
                    "learning_rate": data_setup.LEARNING_RATE,
                    "margin": data_setup.MARGIN,
                    
                    "distance_function": data_setup.distance_function,
                    "norm_type": data_setup.norm_type,
                    "normalizer": normalizer_file,

                    "num_workers": data_setup.NUM_WORKERS,
                    "patience": data_setup.PATIENCE,
                    "min_delta": data_setup.MIN_DELTA,
                    "datafolder": data_setup.data_folder

    }



def main(model):
    #global model
    #ray.init()
    optimizer_str = None

    tuning_params = False


    start_time = time.time()
    train_dataloader, validation_dataloader, test_dataloader = data_setup.create_dataloaders(
                                                                    data_setup.data_folder,
                                                                    data_setup.md_names,
                                                                    data_setup.normalizer,
                                                                    data_setup.dataloader_dir,
                                                                    data_type,
                                                                    train_size=data_setup.TRAIN_SIZE,
                                                                    transform=data_setup.transform, 
                                                                    batch_size=data_setup.BATCH_SIZE, 
                                                                    num_workers=data_setup.NUM_WORKERS,
                                                                    random_seed=data_setup.RANDOM_SEED,
                                                                    distance_function=data_setup.distance_function
                                                                    )
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Time for loading/creating data:", elapsed_time, "seconds")
    
    if tuning_params:
        

        
        # if isinstance(model, model_builder.SiameseNetwork_fcs):
        #     config["fc3"] = ray.tune.choice([128, 1024, 2048, 4096])  # 64, 128
        #     config["fc4"] = ray.tune.choice([128, 512, None])  # 64, 128
        #     config["fc5"] = ray.tune.choice([None])  # 64, 128
            #data_setup.num_samples = 2*data_setup.num_samples
        
        #model = ray.put(model)

        def train_func(config):#, train_dataloader, validation_dataloader):
            #global model
            #ray.tune.utils.wait_for_gpu()
            # model = model_builder.SiameseNetwork_fcs(config["l1"], config["l2"],
            #                 config["l3"], config["l4"],
            #                 config["l5"], config["fc1"], 
            #                 config["fc2"], config["fc3"],
            #                 config["fc4"], config["fc5"])
            # print("inside model before:", model)
            # model = ray.get(model)
            # print("inside model after:", model)
            # if isinstance(model, model_builder.SiameseNetwork):
            model = model_builder.SiameseNetwork(config["l1"], config["l2"],
                        config["l3"], config["l4"],
                        config["l5"], config["fc1"], 
                        config["fc2"])
            # model.l1 = config["l1"]
            # model.l2 = config["l2"]
            # model.l3 = config["l3"]
            # model.l4 = config["l4"]
            # model.l5 = config["l5"]
            # model.fc1 = config["fc1"]
            # model.fc2 = config["fc2"]
            # if isinstance(model, model_builder.SiameseNetwork_fcs):
            #     model.fc3 = config["fc3"]
            #     model.fc4 = config["fc4"]
            #     model.fc5 = config["fc5"]
            
            os.environ["CUDA_VISIBLE_DEVICES"]="2"  # Choose which device to use (astro01 has 4 gpu's)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model.to(device)

            loss_fn = engine.ContrastiveLoss(margin=config["margin"])

            if str(config["optimizer"]) == "ADAM":
                #print("Using ADAM optimizer")
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            elif str(config["optimizer"]) == "SGD":
                #print("Using SDG optimizer")
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
            else:
                print("Optimizer not defined, setting to adam..")
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

            if data_setup.TRAIN_SIZE >= 0.2:
                data_setup.TRAIN_SIZE == 0.2
            
            train_dataloader, validation_dataloader, _ = data_setup.create_dataloaders(
                                                                    data_setup.data_folder,
                                                                    data_setup.md_names,
                                                                    normalizer,
                                                                    data_setup.dataloader_dir,
                                                                    data_type,
                                                                    train_size=data_setup.TRAIN_SIZE, 
                                                                    transform=data_setup.transform, 
                                                                    batch_size=4, 
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
                for batch in validation_dataloader:
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

                    # scaler.scale(torch.mean(validation_loss)).backward(retain_graph=True)
                    # scaler.step(optimizer)
                    # scaler.update()
                validation_loss = np.mean(validation_loss_output)


                val_mse = metrics.mean_squared_error(np.concatenate(truths), np.concatenate(predictions))
                # Report losses
                ray.train.report({"val_loss": validation_loss, "val_mse": val_mse})
                

                # current_memory = torch.cuda.memory_allocated(0)
                # print(f"Total memory usage {current_memory / (1024**2):.2f} MB")

                torch.cuda.empty_cache()
        
        checkpoint_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/checkpoints"
        def tune_main(#config,#train_dataloader, validation_dataloader, 
                      num_samples=10, max_num_epochs=10, cpus=4, gpus_per_trial=1):
    
            config = {
                "l1": ray.tune.choice([32, 64]), #16, 32
                "l2": ray.tune.choice([64, 128]), #32, 64
                "l3": ray.tune.choice([64, 128]), #32, 64, 128
                "l4": ray.tune.choice([64, 256]), #32, 64, 128
                "l5": ray.tune.choice([64, 128]), #64, 128, 256
                "fc1": ray.tune.choice([512, 1024, 2048]), #512, 1024
                "fc2": ray.tune.choice([128, 1024, 2048]), #64, 128
                "lr": ray.tune.choice([8e-5]), #1e-4, 1e-3, 1e-2
                "batch_size": ray.tune.choice([32]), #64
                "margin": ray.tune.choice([1.0]), 
                "optimizer": ray.tune.choice(["ADAM", "SGD"])
            }

            scheduler = ASHAScheduler(
                metric="val_mse",
                mode="min",
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2,
            )
            
            result = ray.tune.run(
                partial(train_func, ),#, train_dataloader, validation_dataloader),
                resources_per_trial={"cpu": cpus, "gpu": gpus_per_trial},
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                local_dir=checkpoint_dir,
            )

            top_trials = result.trials
            top_trials.sort(key=lambda trial: trial.last_result.get("val_mse", float("inf")))

            n_saves = 3


            # Print the configurations and validation losses of the top three trials
            for i, trial in enumerate(top_trials[:n_saves]):
                print(f"Top {i + 1} trial config: {trial.config}")
                print(f"Top {i + 1} trial final validation loss: {trial.last_result['val_loss']}")
                print(f"Top {i + 1} trial final mean squared error: {trial.last_result['val_mse']}")

                print("\n")


            # Save the output to a text file
            tuning_path = "/lustre/astro/antonmol/learning_stuff/siamese_networks/hp_tuning"
            file_path = os.path.join(tuning_path, data_setup.model_name)
            
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            
            "Saving as csv file with pandas"
            params = [trial.config for trial in top_trials]
            df = pd.DataFrame(params)
            df['Validation Loss'] = [trial.last_result['val_loss'] for trial in top_trials]
            df['Mean squared error'] = [trial.last_result['val_mse'] for trial in top_trials]
            csv_file = os.path.join(file_path, f"nsamp{num_samples}_best_hyperparameters.csv")
            df.to_csv(csv_file, sep="\t", index=False)

            
            return top_trials #result
        

        
        os.environ["RAY_DEDUP_LOGS"] = "0"

        #num_samples = num
        #max_num_epochs = 10

        #gpus_per_trial = 0.25
        #cpus = 40 / (1/gpus_per_trial)
        # train_dataset = train_dataloader.dataset
        # validation_dataset = validation_dataloader.dataset
        # subset_indices = torch.randperm(len(train_dataset))[:30]
        # tune_train_dataloader = Subset(train_dataset, subset_indices)
        # tune_validation_dataloader = Subset(validation_dataset, subset_indices)

        #ray.init()
        # print("Model before", model)
        # model = ray.put(model)
        # print("MODEL 2 before", model)
        top_trials = tune_main(num_samples, max_num_epochs, cpus, gpus_per_trial)
        
        #model = ray.get(model)
        print("[INFO] Using best hp configuration params for model")
        data_setup.L1 = top_trials[0].config["l1"]
        data_setup.L2 = top_trials[0].config["l2"]
        data_setup.L3 = top_trials[0].config["l3"]
        data_setup.L4 = top_trials[0].config["l4"]
        data_setup.L5 = top_trials[0].config["l5"]
        data_setup.FC_UNITS1 = top_trials[0].config["fc1"]
        data_setup.FC_UNITS2 = top_trials[0].config["fc2"]
        data_setup.LEARNING_RATE = top_trials[0].config["lr"]
        data_setup.BATCH_SIZE = top_trials[0].config["batch_size"]
        data_setup.MARGIN = top_trials[0].config["margin"]
        # config = {
        #         "l1": ray.tune.choice([32, 64]), #16, 32
        #         "l2": ray.tune.choice([64, 128]), #32, 64
        #         "l3": ray.tune.choice([64, 128]), #32, 64, 128
        #         "l4": ray.tune.choice([64, 256]), #32, 64, 128
        #         "l5": ray.tune.choice([64, 128]), #64, 128, 256
        #         "fc1": ray.tune.choice([512, 1024, 2048]), #512, 1024
        #         "fc2": ray.tune.choice([128, 1024, 2048]), #64, 128
        #         "lr": ray.tune.choice([8e-5]), #1e-4, 1e-3, 1e-2
        #         "batch_size": ray.tune.choice([32]), #64
        #         "margin": ray.tune.choice([1.0]), 
        #         "optimizer": ray.tune.choice(["ADAM", "SGD"])
        #     }

        optimizer_str = top_trials[0].config["optimizer"]
        structure_dict["optimizer"] = optimizer_str
        structure_dict["l1"] = data_setup.L1
        structure_dict["l2"] = data_setup.L2
        structure_dict["l3"] = data_setup.L3
        structure_dict["l4"] = data_setup.L4
        structure_dict["l5"] = data_setup.L5
        structure_dict["fc1"] = data_setup.FC_UNITS1
        structure_dict["fc2"] = data_setup.FC_UNITS2
        structure_dict["lr"] = data_setup.LEARNING_RATE
        structure_dict["batch_size"] = data_setup.BATCH_SIZE
        structure_dict["margin"] = data_setup.MARGIN
        
        

        if "SiameseNetwork_fcs" in data_setup.model_name:  
            model = model_builder.SiameseNetwork(data_setup.L1, data_setup.L2, data_setup.L3, 
                                                 data_setup.L4,data_setup.L5, 
                                                 data_setup.FC_UNITS1, data_setup.FC_UNITS2,
                                                 data_setup.FC_UNITS3, data_setup.FC_UNITS4,
                                                 data_setup.FC_UNITS5)
            print("Network identified as SiameseNetwork_fcs!")
        else:    
            model = model_builder.SiameseNetwork(data_setup.L1, data_setup.L2,
                            data_setup.L3, data_setup.L4,
                            data_setup.L5, data_setup.FC_UNITS1, 
                            data_setup.FC_UNITS2)
        # model.set_parameters(l1=data_setup.L1, l2=data_setup.L1, l3=data_setup.L1, 
        #                      l4=data_setup.L1, l5=data_setup.L1, 
        #                      fc_units1=data_setup.FC_UNITS1, fc_units2=data_setup.FC_UNITS2)
        # model.l1 = data_setup.L1
        # model.l2 = data_setup.L2
        # model.l3 = data_setup.L3
        # model.l4 = data_setup.L4
        # model.l5 = data_setup.L5
        # model.fc_units1 = data_setup.FC_UNITS1
        # model.fc_units2 = data_setup.FC_UNITS2

        if isinstance(model, model_builder.SiameseNetwork_fcs):
            data_setup.FC_UNITS3 = top_trials[0].config["fc3"]
            data_setup.FC_UNITS4 = top_trials[0].config["fc4"]
            data_setup.FC_UNITS5 = top_trials[0].config["fc5"]
            model.fc_units3 = data_setup.FC_UNITS3
            model.fc_units4 = data_setup.FC_UNITS4
            model.fc_units5 = data_setup.FC_UNITS5
            structure_dict["fc3"] = data_setup.FC_UNITS3
            structure_dict["fc4"] = data_setup.FC_UNITS4
            structure_dict["fc5"] = data_setup.FC_UNITS5


        
        # model = model_builder.SiameseNetwork_fcs(data_setup.L1, data_setup.L2, data_setup.L3, data_setup.L4, 
        #                                          data_setup.L5,
        #                                         data_setup.FC_UNITS1, data_setup.FC_UNITS2,
        #                                         data_setup.FC_UNITS3, data_setup.FC_UNITS4,
        #                                         data_setup.FC_UNITS5)



    

    # Print input data shape
    sample_batch = next(iter(train_dataloader))
    batch_shape = sample_batch[0].shape
    print(f"Shape of the batch:")
    print(f"[batch_size, channels, height, width]", {batch_shape})


    ########## Set model, loss_func and optimizer ##########

    

    # img_size = len(sample_batch[0][0][0])
    img_size = data_setup.input_size
    print("IMGSIZE", img_size)
    summary(model, [(1, img_size, img_size),(1, img_size, img_size)], device="cuda")

    loss_fn = engine.ContrastiveLoss(margin=data_setup.MARGIN)
    #loss_fn = nn.CosineEmbeddingLoss(margin=MARGIN)

    if optimizer_str == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=data_setup.LEARNING_RATE
                                    )
    elif optimizer_str == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=data_setup.LEARNING_RATE
                                    )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=data_setup.LEARNING_RATE
                                    )
    

    ########## Add writer ##########

    # Add writer for experiment recording
    log_dir = os.path.join('experiments', data_setup.model_name)
    writer = SummaryWriter(log_dir=log_dir)


    ########## Train! ##########

    model_parameter_df = engine.training(
                            model=model,
                            train_dataloader=train_dataloader,
                            validation_dataloader=validation_dataloader,
                            output_dir=data_setup.output_dir,
                            model_name=data_setup.model_name,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            num_epochs=data_setup.NUM_EPOCHS,
                            device=device,
                            writer=writer,
                            patience=data_setup.PATIENCE,
                            min_delta=data_setup.MIN_DELTA
                            )


    ##########----- Show/save outputs -----##########

    

    structure_df = pd.DataFrame(structure_dict, index=[0]).transpose()
    utils.save_model(model,
                    data_setup.output_dir,
                    data_setup.model_name,
                    structure_df)


    ########## Test! ##########

    engine.testing(model,
                    test_dataloader,
                    device,
                    data_setup.output_dir,
                    data_setup.model_name,
                    )


    ########## Model interpretation ##########

    h5_images = os.path.join(data_setup.output_dir, data_setup.model_name, "batch_images.h5")
    h5_batch_data = os.path.join(data_setup.output_dir, data_setup.model_name, "batch_data.h5")

    attrib_anim = False # If True, save animation of all attributions
    layer = None
    targets = args.targets # Specify which output element of the final FC layers to investigate the model using integrated gradients
    print("targets", targets)

    img1s, img2s = utils.get_batch_data_hdf5(h5_images, ["img1", "img2"])
    if targets > -1 and targets < 998:
        print("Running FC2 method") 
        layer = None
        method = "FC2"
        if attrib_anim == True:
            for n in [0, 1]:
                utils.animate_attr(data_setup.output_dir, data_setup.model_name, n, method, save=True)
        utils.plot_attr(model, 
                        img1s, 
                        img2s, 
                        data_setup.output_dir,
                        data_setup.model_name,
                        device,
                        targets,
                        layer,
                        method)
        
    elif targets == [999]:
        print("Running layer method")
        targets = np.arange(0, L5-1)
        method = "layer"
        layer = model.encoder[0]
        
        utils.plot_attr(model, 
                        img1s, 
                        img2s, 
                        data_setup.output_dir,
                        data_setup.model_name,
                        device,
                        targets,
                        layer,
                        method)




    h5_training_stats = os.path.join(data_setup.output_dir, data_setup.model_name, "training_stats.h5")
    utils.plot_training_evolution(h5_training_stats, data_setup.output_dir, data_setup.model_name)


    if plot_matches == True:
        utils.plot_x_best(h5_batch_data, data_setup.output_dir, data_setup.model_name,
                        sorter="truths", order="high")
        utils.plot_x_best(h5_batch_data, data_setup.output_dir, data_setup.model_name,
                        sorter="truths", order="low")
        utils.plot_x_best(h5_batch_data, data_setup.output_dir, data_setup.model_name,
                        sorter="predictions", order="high")
        utils.plot_x_best(h5_batch_data, data_setup.output_dir, data_setup.model_name,
                        sorter="predictions", order="low")
        
        print("Saved images of 10 best/worst true and predicted matches")


    ##### Matching test #####
    
    test_dataloader_path = os.path.join(data_setup.dataloader_dir, \
            f"tsize{data_setup.TRAIN_SIZE}_bsize{data_setup.BATCH_SIZE}_test_{data_setup.input_size}_{data_setup.distance_function}.pth")

    matching.run(
        data_setup.output_dir,
        data_setup.model_name,
        model,
        test_dataloader,
        test_dataloader_path,
        data_setup.md_names,
        target_normalizer,
        target_projection,
        target_sink,
        target_snapshots,
        data_setup.transform,
        data_setup.BATCH_SIZE,
        device,
        nsamp,
        threshold,
        )
            
    
    if model_parameter_df is not None:
        utils.plot_gradient_evol(model_parameter_df, data_setup.output_dir, data_setup.model_name)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    #torch.cuda.empty_cache()
    main(model)


