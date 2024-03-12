
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gc
from torchinfo import summary


from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import tqdm as tqdm
import itertools


from sklearn.metrics.pairwise import cosine_similarity



""" 
This script includes all the necessary classes and functions needed for training
the Siamese network.

Overview:
    Step 1: Create/load dataset
        class setup_data - 

"""

##### -------------- Step 1: Create/load dataset ----------------

# -------------- Similarity function --------------------


def similarity_function(md1, md2):

    similarity = 0.5

    return similarity



class SetupData(Dataset):

    """
    This class is used to get all images within a target directory.

    Needs to be changed according to new formation for Siamese networks that needs to
    connect images with metadata files.

    * Think about file types
    * Think about naming
    * Thing about how to connect them
        - metadata for two images needs to be compared for each pair in order to estimate
        similarity, so this has to be done when loading the pair.
        - what is better then: 
            * creating complete datasets of pairs, where each pair has a label of similarity or
            * integrating into the training the similarity calculations - this will obviously
            make the training longer and is not preferable.
        conclusion = more advantageous to have a large dataset with labels, to make training faster
        = This function should create that.

    
    """

    def __init__(self, image_folder: str, md_folder: str, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        
        self.md_folder = md_folder
        self.md_paths = [os.path.join(md_folder, f) for f in os.listdir(md_folder)]
        
        self.transform = transform

        # Create all possible pairs of indices
        self.pairs = list(itertools.combinations(range(len(self.image_paths)), 2))
        
    
    def load_data(self, index: int):
        "Opens an image and its metadata via a path and returns them."
        index1, index2 = self.pairs[index]

        # Load image data
        with open(self.image_paths[index1], 'rb') as f:
            img1 = pickle.load(f)
        with open(self.image_paths[index2], 'rb') as f:
            img2 = pickle.load(f)

        
        # Load metadata
        with open(self.md_paths[index1], 'rb') as f:
            md1 = pickle.load(f)
        with open(self.md_paths[index2], 'rb') as f:
            md2 = pickle.load(f)

        return img1, img2, md1, md2

    def __len__(self):
        return len(self.pairs)
        
    def normalize_images(self, img1, img2):
        # Normalize images as in your original code
        minimum = np.min(img1)
        maximum = np.max(img1)
        img1 = np.log10(img1/minimum)/np.log10(maximum/minimum)

        minimum = np.min(img2)
        maximum = np.max(img2)
        img2 = np.log10(img2/minimum)/np.log10(maximum/minimum)

        return img1, img2
    
    def calculate_similarity(self, md1, md2):
        similarity = similarity_function(md1, md2)
        return similarity

    def __getitem__(self, index):
        img1, img2, md1, md2 = self.load_data(index)

        img1, img2 = self.normalize_images(img1, img2)

        similarity_label = self.calculate_similarity(md1, md2)

        # Transform if necessary
        if self.transform:
            return self.transform(img1), self.transform(img2), torch.Tensor([similarity_label])
        else:
            return img1, img2, torch.Tensor([similarity_label])


    

def load_data(image_folder: str, 
              md_folder: str, 
              data_type="pkl", 
              train_size=0.6, 
              transform=None, 
              random_seed=None):
    
    """
    This function creates dataset using the SetupData class.

    Inputs: 
        image_folder - folder for images used in the model.
            OBS: SetupData class will create pairs of images from this folder.
            See class for more information on this.

        md_folder - folder for metadata files used for labelling images.
            See SetupData class for more information on filenaming and setup

        data_type - 

        train_size - Set the fraction of data used for training.
            Uses all available data. The remaining fraction is split evenly for testing
            and validating the model. 
            Set train_size=0.01 for testing the setup. 

        
    
    """

    if data_type == "pkl":

        custom_dataset = SetupData(image_folder=image_folder, 
                               md_folder=md_folder,
                               transform=transform,
                               )
    else:
        custom_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    if random_seed is not None:
        np.random.seed(random_seed)
    
    val_size = 0.5 # 0.5 = Split validation and test sets evenly
    if train_size == 0.01: # For testing set train size = 0.01 and test_size will follow
        test_size = train_size
    else:
        test_size = 1.0-train_size # Use all data
    
    # Split data into training and the rest
    train_dataset, temp_dataset = train_test_split(custom_dataset, 
                                                    train_size=train_size, 
                                                    test_size=test_size,
                                                    random_state=random_seed
    )
    # Split the remainder to equal sized validation and testing sets
    val_dataset, test_dataset = train_test_split(temp_dataset, 
                                                    train_size=val_size, 
                                                    test_size=val_size,
                                                    random_state=random_seed
    )


    
    return train_dataset, val_dataset, test_dataset














# -------------- Create dataloader -----------------------


def create_dataloaders(
    image_folder: str,
    md_folder: str,
    dataloader_dir: str,
    data_type: str,
    train_size: float,
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int,
    random_seed: int,
    ):

    """Creates training and validation DataLoaders.

    Takes in a data path and turns it into training and validation PyTorch Datasets 
    and then into PyTorch DataLoaders. The fraction of data used for training 
    is specified with train_size between 0 and 1 with 1 being all data used for training.

    Args:
    data_dir: Path to data directory.
    data_type: type of data (png, jpg, pkl, ...)
    transform: torchvision transforms to perform on training and validation data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, validation_dataloader).

    Example usage:
        train_dataloader, validation_dataloader = \
        = create_dataloaders(data_dir=path/to/data_dir,
                                data_type='pkl',
                                train_size=0.8,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    
    # Check if saved dataloaders exist
    train_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_batchsize{batch_size}_train.pth")
    validation_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_batchsize{batch_size}_validation.pth")
    test_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_batchsize{batch_size}_test.pth")
 
    if os.path.exists(train_dataloader_path) and os.path.exists(validation_dataloader_path):
        
        # If saved dataloaders exist, load them
        print("\nLoading dataloaders...")
        train_dataloader = torch.load(train_dataloader_path)
        validation_dataloader = torch.load(validation_dataloader_path)
        test_dataloader = torch.load(test_dataloader_path)

        print(f"\n[INFO] Loaded saved dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation batches with batch_size={batch_size}")
        print(f"This gives a total of {len(train_dataloader.dataset)} training images and {len(validation_dataloader.dataset)} validation images")
    else:
        # Create new dataset (takes a while)
        print("Creating dataset...")
        train_data, validation_data, test_data = \
            load_data(image_folder, 
                        md_folder, 
                        data_type, 
                        train_size, 
                        transform, 
                        random_seed
                        )
        
        print(f"Dataset created with {len(train_data)+len(validation_data)} images with shape {train_data[0][0].shape}")

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True, # Set pin_memory to True for GPU acceleration
        )
        validation_dataloader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"\n[INFO] Created dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation batches with batch_size={batch_size}")
        print(f"This gives a total of {len(train_data)} training images and {len(validation_data)} validation images")
        print("Saving dataloaders...")
        # Check if the folder exists, create it if it doesn't
        if not os.path.exists(dataloader_dir):
            os.makedirs(dataloader_dir)

        # Save train and validation dataloaders
        torch.save(train_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_batchsize{batch_size}_train.pth"))
        torch.save(validation_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_batchsize{batch_size}_validation.pth"))
        torch.save(test_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_batchsize{batch_size}_test.pth"))
        print("Dataloaders saved at", dataloader_dir)

    return train_dataloader, validation_dataloader, test_dataloader













##### -------------- Step 2: Define model --------------------

class SiameseNetwork(nn.Module):

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork, self).__init__()

        "The architecture of the Siamese Network"

        # N, 1, 800, 800

        # CNN layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=2, padding=1), # -> N, l1, 400, 400
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=2, padding=1), # -> N, l2, 200, 200
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(l2, l3, 3, stride=2, padding=1), # -> N, l3, 100, 100
            nn.ReLU(),
            nn.Conv2d(l3, l4, 7, stride=5, padding=1), # -> N, l4, 20, 20 
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=2, padding=1), # -> N, l5, 10, 10 
            nn.ReLU(),
        )

        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*10*10, fc_units1),
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        # Similarity function
        self.distance_metric = nn.PairwiseDistance()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        similarity = self.distance_metric(output1, output2)
        similarity = similarity.to(self.device).float()
        return output1, output2, similarity








##### ------------- Step 3: Define training function ------------

# Define the Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2) # THINK ABOUT SIMILARITY MEASURE OF FC LAYERS

        # OBS - REWRITE THIS?
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive



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



def training(model: nn.Module,
             train_dataloader: DataLoader,
             validation_dataloader: DataLoader,
             loss_fn: nn.Module,
             optimizer: optim.Optimizer,
             num_epochs: int,
             device: torch.device,
             writer: SummaryWriter,
             patience: int,
             min_delta: float):
    
    print("[INFO] Training started")
    encoded_outputs = []
    
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    scaler = torch.cuda.amp.GradScaler() 
    
    # ------TRAINING------
    for epoch in range(num_epochs):
        model.train()
        training_loss_output = []


        for batch in train_dataloader:
            img1, img2 = batch[0], batch[1]
            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            sim_score = batch[2].to(device).float()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output1, output2, similarity = model(img1, img2) 
                assert output1.dtype is torch.float16
                assert output2.dtype is torch.float16
                #assert similarity.dtype is torch.float16
                training_loss = loss_fn(output1, output2, sim_score) 
                assert training_loss.dtype is torch.float32

            scaler.scale(training_loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

            # Save outputs!
            encoded_outputs.append((output1, output2, similarity))
            training_loss_output.append(training_loss.item())

        training_loss = np.mean(training_loss_output) # Getting the mean loss for the batch
        

        if writer:
            # Add results to SummaryWriter
            writer.add_scalar(tag="Training loss", 
                            scalar_value=np.mean(training_loss_output),
                            global_step=epoch)
            
            writer.add_graph(model=model, 
                        # Pass in an example input
                        input_to_model=torch.randn(32, 1, 224, 224).to(device))
            # Close the writer
            writer.close()
        
        model.eval()  # Set the model to evaluation mode
        validation_loss_output = []

        # with torch.no_grad():
        #     for batch in validation_dataloader:
        #         val_img1, val_img2 = batch[0], batch[1]
        #         val_img1 = val_img1.to(device).float()  # Convert to float
        #         val_img2 = val_img2.to(device).float()
        #         val_output1, val_output2, val_similarity = model(val_img1, val_img2)
        #         validation_loss = loss_fn(val_output1, val_output2)
        #         validation_loss_output.append(validation_loss.item())

        for batch in validation_dataloader:
            val_img1, val_img2 = batch[0], batch[1]
            val_img1 = val_img1.to(device).float()
            val_img2 = val_img2.to(device).float()
            val_sim_score = batch[2].to(device).float()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                val_output1, val_output2, val_similarity = model(val_img1, val_img2) 
                assert val_output1.dtype is torch.float16
                assert val_output2.dtype is torch.float16
                #assert val_similarity.dtype is torch.float16
                validation_loss = loss_fn(val_output1, val_output2, val_sim_score) 
                assert validation_loss.dtype is torch.float32
                
                validation_loss_output.append(validation_loss.item())


            scaler.scale(validation_loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

        validation_loss = np.mean(validation_loss_output)
        
        print(f'Epoch:{epoch+1}, Training Loss: {training_loss:.6f}, Validation Loss: {validation_loss:.6f}')
        
        if writer:
            # Add validation loss to SummaryWriter
            writer.add_scalar(tag="Validation Loss",
                            scalar_value=np.mean(validation_loss_output),
                            global_step=epoch)
        
        early_stopping(validation_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break 

        if writer:
            writer.close()

        # Calculate memory usage after each epoch
        current_memory = torch.cuda.memory_allocated(0)

        print(f"Total memory usage {current_memory / (1024**2):.2f} MB")

        # Save encoded_outputs and decoded_outputs at the end of each epoch (OBS: Increases runtime)
        #torch.save(encoded_outputs, f'encoded_outputs_epoch_{epoch}.pt')
        #torch.save(decoded_outputs, f'decoded_outputs_epoch_{epoch}.pt')

        if epoch < num_epochs-1:
            # Clear the lists to free up memory for the next epoch
            encoded_outputs = []
            #decoded_outputs = []

        #gc.collect()
        #stats = gc.get_stats()
        torch.cuda.empty_cache()
    print("[INFO] Training done")
    return encoded_outputs


def create_writer(experiment_name: str, 
                  model_name: str,
                  timestamp: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

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





##### ----------------- Step 4: Utilities --------------------





def save_model(model: torch.nn.Module,
               output_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create folder within output directory for storing model
    model_dir_path = os.path.join(output_dir, model_name)
    #/lustre/astro/antonmol/learning_stuff/autoencoder/encoder_modular/outputs/test_autoencoder_model
    output_dir_path = Path(model_dir_path)
    output_dir_path.mkdir(parents=True,
                         exist_ok=True)
    
    # Create model save path
    #assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

    # Save the model state_dict()"
    print(f"[INFO] Saving model to: {output_dir_path}")
    model_savename = "{}/{}.pth".format(output_dir_path, model_name)  # Name of saved model
    torch.save(obj=model.state_dict(),
             f=model_savename)





def show_output(encoded_outputs,
                train_dataloader,
                output_dir: str,
                model_name: str,):
    """

    """
    # Encoded outputs: (batch, img1, img2, similarity)
    first_batch = encoded_outputs[0] # Batch nr.
    img1 = first_batch[0][0].detach().cpu().numpy()
    img2 = first_batch[1][0].detach().cpu().numpy()
    sim_score = first_batch[2][0].detach().cpu().numpy()

    # Input images for comparisons
    first_batch = next(iter(train_dataloader))
    input_img1 = first_batch[0][0][0].detach().cpu().numpy()
    input_img2 = first_batch[1][0][0].detach().cpu().numpy()
    ground_truth = first_batch[2].detach().cpu().numpy()
    print("Ground truth:", ground_truth)
    plt.figure(figsize=(10,8))

    plt.subplot2grid((3, 2), (0, 0)); plt.imshow(input_img1, label="Input image 1"); plt.colorbar()
    plt.subplot2grid((3, 2), (0, 1)); plt.imshow(input_img2, label="Input image 2"); plt.colorbar()

    plt.subplot2grid((3, 2), (1, 0)); plt.plot(abs(img1), '.', label="Img1"); plt.legend()
    plt.subplot2grid((3, 2), (1, 1)); plt.plot(abs(img2), '.', label="Img2"); plt.legend()
    plt.subplot2grid((3, 2), (2, 0), colspan=2); plt.plot(abs(img1-img2), '.', label="Abs(img1-img2)");
    plt.legend(); plt.ylim(0, np.mean(abs(img1)))
    #plt.title("Similarity score:", sim_score)
    
    plt.suptitle(f"Similarity score: {sim_score}", fontsize=16)

    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Create target directory
    #/lustre/astro/antonmol/learning_stuff/autoencoder/encoder_modular/outputs/test_autoencoder_model
    save_name = os.path.join(output_dir, model_name, "images")
    output_dir_path = Path(save_name)
    if not os.path.exists(output_dir_path):
        output_dir_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_name}/similarities.png")