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
import data_setup, model_builder, engine, utils, matching, tuning, tuning2
import matplotlib.pyplot as plt
import argparse
import ray
from sklearn import metrics
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import json


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
parser.add_argument("--data_type", type=str, default="krisha", help="Type of data. pkl=old Vito data, krisha=our data")

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
matching_test = True
pre_train_interpretation = False

data_type = args.data_type


########## Directory params ##########
data_setup.output_dir = args.output_dir
data_setup.model_name = args.model_name
data_setup.data_folder = args.data_folder
data_setup.dataloader_dir = args.dataloader_dir

# Create folders for saving output and model
if not os.path.exists(data_setup.output_dir):
    os.mkdir(data_setup.output_dir)


########## Metadata params ##########
data_setup.distance_function = "cosine"  # cosine, euclid
data_setup.norm_type = "z_score" # minmax, z_score
#DISTANCE_FUNCTION = data_setup.distance_function  # cosine, euclid

if data_type == "pkl":
    data_setup.normalizer_file = "metadata_stats_sink49.csv"
elif data_type == "krisha":
    data_setup.md_names = ["m_env", "ar", "temp", "m", "ds", "lum", "t"]
    sink = 24
    data_setup.normalizer_file = f"metadata_stats_sink{sink}_krisha.csv"
normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/{data_setup.normalizer_file}", index_col=0)
data_setup.normalizer = normalizer

########## Tuning params ##########
num_samples = args.num_samples
max_num_epochs = args.max_num_epochs
gpus_per_trial = 0.25
cpus = 40 / (1/gpus_per_trial)



########## Create/load dataset ##########
data_setup.input_size = 512 # Change input image size if using cropping
print("Input image size:", data_setup.input_size)

# Setup target device
os.environ["CUDA_VISIBLE_DEVICES"] = data_setup.device_nr  # Choose which device to use (astro01 has 4 gpu's)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available
print(f"\nUsing {device} as device")
print("Current device:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Device name", torch.cuda.get_device_name(torch.cuda.current_device()))

# Initialize model for training (will be overwritten by experiments.py)
model = model_builder.SiameseNetwork(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2)


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
                    "normalizer": data_setup.normalizer_file,

                    "num_workers": data_setup.NUM_WORKERS,
                    "patience": data_setup.PATIENCE,
                    "min_delta": data_setup.MIN_DELTA,
                    "datafolder": str(data_setup.data_folder)

    }

tuning_params = False

def main(model, model_name):
    print("[INFO] Running main training setup for model:")
    data_setup.model_name = model_name
    print(data_setup.model_name)
    # Matching params
    nsamp = 50
    threshold = None
    
    target_projection = "3"
    if data_type == "pkl":
        target_sink = "164"
        target_normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_sink{target_sink}.csv", index_col=0)
        target_snapshots = ["000454"]#, "000578", "000638", "000690"]
    elif data_type == "krisha":
        target_sink = "178"
        target_normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_sink{target_sink}_krisha.csv", index_col=0)
        target_snapshots = ["1000"]#, "000578", "000638", "000690"]
    #global model
    #ray.init()
    optimizer_str = None

    #tuning_params = True

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
                                                                    distance_function=data_setup.distance_function,
                                                                    save_dataset=data_setup.save_dataset,
                                                                    )
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Time for loading/creating data:", elapsed_time, "seconds")
    

    if tuning_params:
        
        os.environ["RAY_DEDUP_LOGS"] = "1"
        top_trials = tuning2.tune_main(data_setup.model_name, data_setup.md_names, num_samples, max_num_epochs, cpus, gpus_per_trial)
        
        print("[INFO] Using best hp configuration params for model")
        data_setup.L1 = top_trials[0].config["l1"]
        data_setup.L2 = top_trials[0].config["l2"]
        data_setup.L3 = top_trials[0].config["l3"]
        data_setup.L4 = top_trials[0].config["l4"]
        data_setup.L5 = top_trials[0].config["l5"]
        data_setup.FC_UNITS1 = top_trials[0].config["fc1"]
        data_setup.FC_UNITS2 = top_trials[0].config["fc2"]
        data_setup.LEARNING_RATE = top_trials[0].config["lr"]
        data_setup.MARGIN = top_trials[0].config["margin"]

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
        structure_dict["margin"] = data_setup.MARGIN
        
        if "SiameseNetwork_batchsize" in data_setup.model_name:
            model = model_builder.SiameseNetwork(data_setup.L1, data_setup.L2,
                        data_setup.L3, data_setup.L4,
                        data_setup.L5, data_setup.FC_UNITS1, 
                        data_setup.FC_UNITS2)
            print("UPDATED MODEL AFTER TUNING")
        if "SiameseNetwork_he_init_batchsize" in data_setup.model_name:
            model = model_builder.SiameseNetwork_he_init(data_setup.L1, data_setup.L2,
                        data_setup.L3, data_setup.L4,
                        data_setup.L5, data_setup.FC_UNITS1, 
                        data_setup.FC_UNITS2)
        if "SiameseNetwork_he_init_batchnorm_batchsize" in data_setup.model_name:
            model = model_builder.SiameseNetwork_he_init_batchnorm(data_setup.L1, data_setup.L2,
                        data_setup.L3, data_setup.L4,
                        data_setup.L5, data_setup.FC_UNITS1, 
                        data_setup.FC_UNITS2)
        if "SiameseNetwork_he_init_batchnorm_ELU_batchsize" in data_setup.model_name:
            model = model_builder.SiameseNetwork_he_init_batchnorm_ELU(data_setup.L1, data_setup.L2,
                        data_setup.L3, data_setup.L4,
                        data_setup.L5, data_setup.FC_UNITS1, 
                        data_setup.FC_UNITS2)
        if "SiameseNetwork_fcs" in data_setup.model_name:
            data_setup.FC_UNITS3 = top_trials[0].config["fc3"]
            data_setup.FC_UNITS4 = top_trials[0].config["fc4"]
            data_setup.FC_UNITS5 = top_trials[0].config["fc5"]
            model.fc_units3 = data_setup.FC_UNITS3
            model.fc_units4 = data_setup.FC_UNITS4
            model.fc_units5 = data_setup.FC_UNITS5
            structure_dict["fc3"] = data_setup.FC_UNITS3
            structure_dict["fc4"] = data_setup.FC_UNITS4
            structure_dict["fc5"] = data_setup.FC_UNITS5

            model = model_builder.SiameseNetwork_fcs(data_setup.L1, data_setup.L2,
                        data_setup.L3, data_setup.L4,
                        data_setup.L5, data_setup.FC_UNITS1, 
                        data_setup.FC_UNITS2, data_setup.FC_UNITS3,
                        data_setup.FC_UNITS4, data_setup.FC_UNITS5)
                
    # Print input data shape
    sample_batch = next(iter(train_dataloader))
    batch_shape = sample_batch[0].shape
    print(f"Shape of the batch:")
    print(f"[batch_size, channels, height, width]", batch_shape)
    img1s, img2s = sample_batch[0], sample_batch[1]

    ### Pre-training model interpretation ###
    if pre_train_interpretation == True:
        print(f"\n[INFO] Model interpretation started for untrained model.")
        model_dir = os.path.join(data_setup.output_dir, data_setup.model_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        image_dir = os.path.join(model_dir, "images")
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        
        target_types = [[1], [data_setup.L5-1], [np.arange(0, data_setup.L5-1)]]
        plot_imgs = [img1s[0][0], img2s[0][0]]
        # For output nodes
        for target_type in target_types:
            targets = target_type
            utils.plot_attr_homemade(model, img1s, img2s, plot_imgs, data_setup.output_dir, data_setup.model_name, device,
                                targets, layer=None, method="FC2", sign="all", summing=True, order="untrained")
        
        # For layer(s)
        layer = model.encoder[0]
        utils.plot_attr_homemade(model, img1s, img2s, plot_imgs, data_setup.output_dir, data_setup.model_name, device,
                                targets, layer=layer, method="layer", sign="all", summing=True, order="untrained")
    else:
        print("[INFO] No pre-trained model interpretation is done.")

    ########## Set model, loss_func and optimizer ##########
    # img_size = len(sample_batch[0][0][0])
    img_size = data_setup.input_size
    summary(model, [(1, img_size, img_size),(1, img_size, img_size)], device="cuda")

    loss_fn = engine.ContrastiveLoss(margin=data_setup.MARGIN)

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
        print("Using Adam optimizer")
    

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
    print(structure_dict)
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
    print(f"\n[INFO] Model interpretation started for trained model.")
    attrib_anim = False # If True, save animation of all attributions
    h5_images = os.path.join(data_setup.output_dir, data_setup.model_name, "batch_images.h5")
    h5_batch_data = os.path.join(data_setup.output_dir, data_setup.model_name, "batch_data.h5")
    
    # For output nodes
    target_types = [[1], [data_setup.L5], np.arange(0, data_setup.L5-1)]
    plot_imgs = [img1s[0][0], img2s[0][0]]
    for target_type in target_types:
        targets = target_type
        try:
            utils.plot_attr_homemade(model, img1s, img2s, plot_imgs, data_setup.output_dir, data_setup.model_name, device,
                                targets, layer=None, method="FC2", sign="all", summing=True, order="trained")
        except Exception as e:
            print("There was an error in creating attributions for untrained model:")
            print(e)

    # For layer(s)
    layer = model.encoder[0]
    try:
        utils.plot_attr_homemade(model, img1s, img2s, plot_imgs, data_setup.output_dir, data_setup.model_name, device,
                                targets, layer=layer, method="layer", sign="all", summing=True, order="trained")
    except Exception as e:
        print("There was an error in creating attributions for untrained model:")
        print(e)

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
    if matching_test == True:
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
            data_type=data_type
            )
            
    
    if model_parameter_df is not None:
        utils.plot_gradient_evol(model_parameter_df, data_setup.output_dir, data_setup.model_name)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main(model, data_setup.model_name)


