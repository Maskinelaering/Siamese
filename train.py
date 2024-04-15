if __name__ == "__main__":
    print("Loading modules...")

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
import data_setup, model_builder, engine, utils, matching
import matplotlib.pyplot as plt
import argparse

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
TRAIN_SIZE = 0.01
BATCH_SIZE = 4
NUM_WORKERS = 1
NUM_EPOCHS = 10
MARGIN = 0.9
PATIENCE = 5
MIN_DELTA = 1e-3
LR = 1e-4
L1 = 32
L2 = 32
L3 = 128
L4 = 64
L5 = 128
FC1 = 512
FC2 = 128
targets = -1
RANDOM_SEED = None
output_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs"
model_name = "just_testing"
data_folder = "/lustre/astro/antonmol/atm_new_dataset_sink49"
dataloader_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders"

plot_matches = True
device_nr = "2"     
data_type = 'pkl'

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Train a Siamese Network")

    parser.add_argument("--train_size", type=float, default=0.01, help="Train size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="...")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train model over")
    parser.add_argument("--margin", type=float, default=1.0, help="Least distance between similar and non-similar image pairs. Criteria for contrastiv loss function")
    parser.add_argument("--patience", type=int, default=5, help="For early-stopping: Number of consecutive epochs without improvement to val_loss for early stopping.")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="For early-stopping: Minimum improvement in val_loss for early stopping.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--l1", type=int, default=32, help="Number of channels at layers with l1 parameter. Check model for clarification.")
    parser.add_argument("--l2", type=int, default=32, help="Number of channels at layers with l2 parameter. Check model for clarification.")
    parser.add_argument("--l3", type=int, default=128, help="Number of channels at layers with l3 parameter. Check model for clarification.")
    parser.add_argument("--l4", type=int, default=64, help="Number of channels at layers with l4 parameter. Check model for clarification.")
    parser.add_argument("--l5", type=int, default=128, help="Number of channels at layers with l5 parameter. Check model for clarification.")
    parser.add_argument("--fc1", type=int, default=512, help="Number of nodes in first FC layer. Check model for clarification.")
    parser.add_argument("--fc2", type=int, default=128, help="Number of nodes in second FC layer, final output. Check model for clarification.")
    parser.add_argument("--targets", type=utils.parse_targets, default=-1, help="The output node from the final FC layer to investigate in model interpretation")

    parser.add_argument("--output_dir", type=str, default="/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs", help="Path to folder for storing outputs such as trained model, plots and latent feature representations (not dataloaders)")
    parser.add_argument("--model_name", type=str, default="just_testing", help="Model name")
    parser.add_argument("--data_folder", type=str, default="/lustre/astro/antonmol/atm_new_dataset_sink49", help="Path to datafolder")
    parser.add_argument("--dataloader_dir", type=str, default="/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders", help="Path to folder for storing dataloaders")

    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for data(?)")

    args = parser.parse_args()

    ########## Learning params ##########
    BATCH_SIZE = args.batch_size  # Number of inputs in each batch for *multi-core processing*?
    TRAIN_SIZE = args.train_size # Fraction of data used for training. The remainder is split evenly for validation and testing
    NUM_WORKERS = args.num_workers # 

    NUM_EPOCHS = args.num_epochs # Number of iterations. Model sees same input and updates its weights each iteration.
    MARGIN = args.margin    # Minimum distance
    LEARNING_RATE = args.lr
    PATIENCE = args.patience    # For EarlyStopping. Number of consecutive epochs without MIN_DELTA improvement to val_loss before stopping early
    MIN_DELTA = args.min_delta # For EarlyStopping. Minimal improvement for loss needed to not trigger early stopping.
    RANDOM_SEED = args.random_seed

    L1 = args.l1
    L2 = args.l2
    L3 = args.l3
    L4 = args.l4
    L5 = args.l5
    FC_UNITS1 = args.fc1
    FC_UNITS2 = args.fc2

    ########## Directory params ##########

    output_dir = args.output_dir
    model_name = args.model_name
    data_folder = args.data_folder
    dataloader_dir = args.dataloader_dir

    # Create folders for saving output and model
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    ########## Metadata params ##########

    md_names = ["ar_p", "ar_s", "mih", "m", "ds", "sep", "t"]

    data_setup.distance_function = "cosine"  # cosine, euclid
    data_setup.norm_type = "z_score" # minmax, z_score
    DISTANCE_FUNCTION = data_setup.distance_function  # cosine, euclid

    normalizer_file = "metadata_stats_sink49.csv"
    normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/{normalizer_file}", index_col=0)

    ########## Create/load dataset ##########

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.CenterCrop(400), # Crop image to 400x400 pixels 
        #transforms.RandomRotation(45) # Randomly rotates between min/max of specified value
        # NOTE: Remember to change model_builder.input_size accordingly, if changing size of image by cropping

        # transforms.RandomErasing(),
        # Add transform here
    ])

    data_setup.input_size = 400 # Change input image size if using cropping
    print("Input image size:", data_setup.input_size)

    train_dataloader, validation_dataloader, test_dataloader = data_setup.create_dataloaders(
                                                                data_folder,
                                                                md_names,
                                                                normalizer,
                                                                dataloader_dir,
                                                                data_type,
                                                                train_size=TRAIN_SIZE,
                                                                transform=transform, 
                                                                batch_size=BATCH_SIZE, 
                                                                num_workers=NUM_WORKERS,
                                                                random_seed=RANDOM_SEED,
                                                                distance_function=DISTANCE_FUNCTION
                                                                )


    # Setup target device
    os.environ["CUDA_VISIBLE_DEVICES"] = device_nr  # Choose which device to use (astro01 has 4 gpu's)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available
    print(f"\nUsing {device} as device")
    print("Current device:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Device name", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Print input data shape
    sample_batch = next(iter(train_dataloader))
    batch_shape = sample_batch[0].shape
    print(f"Shape of the batch:")
    print(f"[batch_size, channels, height, width]", {batch_shape})




    ########## Set model, loss_func and optimizer ##########

    model = model_builder.SiameseNetwork_he_init_batchnorm(l1=L1, l2=L2, l3=L3, l4=L4, l5=L5, 
                                        fc_units1=FC_UNITS1, 
                                        fc_units2=FC_UNITS2)

    img_size = len(sample_batch[0][0][0])
    print("IMGSIZE", img_size)
    summary(model, [(1, img_size, img_size),(1, img_size, img_size)], device="cuda")

    loss_fn = engine.ContrastiveLoss(margin=MARGIN)
    #loss_fn = nn.CosineEmbeddingLoss(margin=MARGIN)


    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE
                                )
    

    ########## Add writer ##########

    # Add writer for experiment recording
    log_dir = os.path.join('experiments', model_name)
    writer = SummaryWriter(log_dir=log_dir)


    ########## Train! ##########

    engine.training(model=model,
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader,
                    output_dir=output_dir,
                    model_name=model_name,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    num_epochs=NUM_EPOCHS,
                    device=device,
                    writer=writer,
                    patience=PATIENCE,
                    min_delta=MIN_DELTA)


    ##########----- Show/save outputs -----##########

    structure_dict = {"model_name": model_name,
                    "train_size": TRAIN_SIZE, 
                    "input_size": data_setup.input_size,
                    "batch_size": BATCH_SIZE,
                    "l1": L1, 
                    "l2": L2, 
                    "l3": L3, 
                    "l4": L4, 
                    "l5": L5, 
                    "fc1": FC_UNITS1, 
                    "fc2": FC_UNITS2,
                    "learning_rate": LEARNING_RATE,
                    "margin": MARGIN,
                    
                    "distance_function": DISTANCE_FUNCTION,
                    "norm_type": data_setup.norm_type,
                    "normalizer": normalizer_file,

                    "num_workers": NUM_WORKERS,
                    "patience": PATIENCE,
                    "min_delta": MIN_DELTA,
                    "datafolder": data_folder

    }

    structure_df = pd.DataFrame(structure_dict, index=[0]).transpose()
    utils.save_model(model,
                    output_dir,
                    model_name,
                    structure_df)


    ########## Test! ##########

    engine.testing(model,
                    test_dataloader,
                    device,
                    output_dir,
                    model_name,
                    )


    ########## Model interpretation ##########

    h5_filename = os.path.join(output_dir, model_name, "batch_data.h5")
    attrib_anim = False # If True, save animation of all attributions
    layer = None
    targets = args.targets # Specify which output element of the final FC layers to investigate the model using integrated gradients
    print("targets", targets)

    img1s, img2s = utils.get_batch_data_hdf5(h5_filename, ["img1", "img2"])
    if targets > -1 and targets < 998:
        print("Running FC2 method") 
        layer = None
        method = "FC2"
        if attrib_anim == True:
            for n in [0, 1]:
                utils.animate_attr(output_dir, model_name, n, save=True)
        utils.plot_attr(model, 
                        img1s, 
                        img2s, 
                        output_dir,
                        model_name,
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
                        output_dir,
                        model_name,
                        device,
                        targets,
                        layer,
                        method)




    h5_training_stats = os.path.join(output_dir, model_name, "training_stats.h5")
    utils.plot_training_evolution(h5_training_stats, output_dir, model_name)


    if plot_matches == True:
        utils.plot_x_best(h5_filename, output_dir, model_name,
                        sorter="truths", order="high")
        utils.plot_x_best(h5_filename, output_dir, model_name,
                        sorter="truths", order="low")
        utils.plot_x_best(h5_filename, output_dir, model_name,
                        sorter="predictions", order="high")
        utils.plot_x_best(h5_filename, output_dir, model_name,
                        sorter="predictions", order="low")
        
        print("Saved images of 10 best/worst true and predicted matches")


    ##### Matching test #####
    nsamp = 0.8
    target_sink = "164"
    target_projection = "3"
    target_normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_sink{target_sink}.csv", index_col=0)

    target_snapshots = ["000454"]#, "000578", "000638", "000690"]
    
    test_dataloader_path = os.path.join(dataloader_dir, \
            f"tsize{TRAIN_SIZE}_bsize{BATCH_SIZE}_test_{data_setup.input_size}_{data_setup.distance_function}.pth")

    matching.run(
        output_dir,
        model_name,
        model,
        test_dataloader,
        test_dataloader_path,
        md_names,
        target_normalizer,
        target_projection,
        target_sink,
        target_snapshots,
        transform,
        BATCH_SIZE,
        device,
        nsamp,
        )
            





