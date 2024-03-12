
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pickle
import os
import tqdm as tqdm
import itertools
import pandas as pd
from sklearn import metrics
import random
import math

"""
This script is for creating PyTorch dataloaders. 

NOTE: Dataloaders are saved and will not be over-written. Any changes made in this document 
will not be applied to existing dataloaders. When making changes, consider deleting existing dataloaders
to be certain that all changes are applied.

"""
input_size = 400

test_similarity = None

distance_function = "cosine"  # cosine, euclid
norm_type = "z_score" # minmax, z_score

##### -------------- Step 1: Create/load dataset ----------------


# min, max, mean and std values computed for each metadata parameter
ar_s_stats = [4.920575656729093e-14 , 4.883139373160987e-13 , 1.5998683502328056e-13 , 5.853457864542639e-14] #[min, max, mean, std]
ar_p_stats = [1.7612091524905243e-06 , 4.5039630181773166e-05 , 1.1237145560259502e-05 , 7.111484935601497e-06]
mih_stats = [1.1012342557696856e+30 , 1.175988350929672e+32 , 2.262783690105513e+31 , 2.440384162054931e+31]
m_stats = [4.747706220749683e-05 , 0.0006432782806819867 , 0.000423637317739924 , 0.00015939441534300718]
ds_stats = [0.0 , 458.1103546350497 , 150.011442111849 , 94.89227491504303]
sep_stats = [225.1923007248629 , 5779.502318739806 , 1392.750789695731 , 1619.5235585032883]
t_stats = [4.15463518978504 , 166.8422452216837 , 85.73724726640633 , 48.529424497343555]


# -------------- Metadata extraction ----------------

def metadata_normalization(md_paths):
    
    "Calculate minimum and maximum for each parameter to normalize it"
    
    print("Normalizing metadata...")
    metadata_files = []
    ar_p = []
    ar_s = []
    mih = []
    m = []
    ds = []
    sep = []
    t = []

    for subfolder in md_paths:
        with open(subfolder, 'rb') as f:
            data = pickle.load(f)
            metadata_files.append(data)

            S = data
            p = data["primary sink tag"]

            md_p = {"ar_p": S["accretion_rate_primary"], "ar_s": S["accretion_rate_secondary"].value,
                    "mih" : S["mass_in_100"] , "m": S["rsink file"]["m"][p], "ds": S["disk size"],
                    "sep": S["separation"].value, "t": S["t_after_formation"].value,
                    }
            
            MD = md_p
            ar_p.append(MD['ar_p'])
            ar_s.append(MD['ar_s'])
            mih.append(MD['mih'])
            m.append(MD['m'])
            ds.append(MD['ds'])
            sep.append(MD['sep'])
            t.append(MD['t'])
            

    all_metadata = [ar_p, ar_s, mih, m, ds, sep, t]

    print(f"Gathered all {len(ar_p)} params")
    
    return all_metadata


def get_metadata(md, filetype="dataframe", norm_type="z_score"):
    
    S = md
    p = md["primary sink tag"]

    # Calculate orbital phase here..

    # G = 100
    # v = S['u'][i1]
    # r = S['x'][i1]
    # M1, M2 = S['m'][i1], S['m'][i2]
    # a = ((2 / r) - (v**2 / (G*(M1 + M2))))**(-1)
    # e = math.sqrt((1-r/a))
    # OP = math.acos((a * (1-e**2) / r) - 1)


    if norm_type == "minmax":
        
        md_p = {"ar_p": (S['accretion_rate_primary'] - ar_p_stats[0]) / (ar_p_stats[1] - ar_p_stats[0]), 
                "ar_s": ((S['accretion_rate_secondary'].value) - ar_s_stats[0]) / (ar_s_stats[1] - ar_s_stats[0]), 
                "mih": ((S['mass_in_100']) - mih_stats[0]) / (mih_stats[1] - mih_stats[0]),
                "m": ((S["rsink file"]["m"][p]) - m_stats[0]) / (m_stats[1] - m_stats[0]), 
                "ds": ((S['disk size']) - ds_stats[0]) / (ds_stats[1] - ds_stats[0]), 
                "sep": ((S['separation'].value) - sep_stats[0]) / (sep_stats[1] - sep_stats[0]),
                "t": ((S['t_after_formation'].value) - t_stats[0]) / (t_stats[1] - t_stats[0]), 
                                }
    elif norm_type == "z_score":
    

        md_p = {"ar_p": (S["accretion_rate_primary"] - ar_p_stats[2]) / (ar_p_stats[3]), 
                "ar_s": (S["accretion_rate_secondary"].value - ar_p_stats[2]) / (ar_p_stats[3]),
                "mih" : (S["mass_in_100"] - mih_stats[2]) / (mih_stats[3]),  
                "m": (S["rsink file"]["m"][p] - m_stats[2]) / (m_stats[3]), 
                "ds": (S["disk size"] - ds_stats[2]) / (ds_stats[3]),
                "sep": (S["separation"].value - sep_stats[2]) / (sep_stats[3]), 
                "t": (S["t_after_formation"].value - t_stats[2]) / (t_stats[3]),
                    }

    
    if filetype == "dataframe":
        primary_md = pd.DataFrame(md_p, index=["Primary"])
        #secondary_md = pd.DataFrame(md_s, index=["Secondary"])
        #metadata = pd.concat([primary_md, secondary_md])
    
    return primary_md

# -------------- Similarity function --------------------
def similarity_function(md1, md2, distance_function="euclid"):

    MD1 = get_metadata(md1, filetype="dataframe", norm_type=norm_type)
    MD2 = get_metadata(md2, filetype="dataframe", norm_type=norm_type)
    
    MD1 = np.array(MD1)
    MD2 = np.array(MD2)
   
    MD1_tensor = torch.tensor(np.array(MD1))  # Convert MD1 to a PyTorch tensor
    MD2_tensor = torch.tensor(np.array(MD2))

    if distance_function == "cosine":
        truth = F.cosine_similarity(MD1_tensor, MD2_tensor)


        # truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
        # truth = 1-truth
        # x = 5 * (truth - 0.5)
        # truth = 1 / (1 + np.exp(-x))
        
        #print("truthshape", truth.shape)
        #print("TRUTH", truth)

        truth = truth.ravel()[0]

    if distance_function == "euclid":
        #truth = metrics.euclidean_distances(MD1_log, MD2_log)
        #truth = np.sqrt(np.sum((MD1 - MD2)**2))
        truth = F.pairwise_distance(MD1_tensor, MD2_tensor)
        #truth = torch.sigmoid(-truth_tensor)
        #truth = truth_tensor.numpy()
        #truth = truth.ravel()[0]
        
        #print("TRUTH", truth)
        

    return truth



class SetupData(Dataset):

    """
    This class is used to get all images within a target directory.


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

    def __init__(self, data_folder: str, transform=None, fraction=0.01, distance_function="cosine"):
        self.image_folder = data_folder
        self.image_paths = []
        self.md_paths = []
        self.distance_function = distance_function
        
        # Iterate over subfolders within data_folder
        all_subfolders = [subfolder for subfolder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, subfolder))]

        size = int(fraction * len(all_subfolders))
        selected_subfolders = random.sample(all_subfolders, size)

        for subfolder in selected_subfolders:
            subfolder_path = os.path.join(data_folder, subfolder)

            if os.path.isdir(subfolder_path):
                image_paths_subfolder = [os.path.join(data_folder, subfolder_path, f) for f in os.listdir(subfolder_path) if "projection" in f]
                md_paths_subfolder = [os.path.join(data_folder, subfolder, f) for f in os.listdir(subfolder_path) if "characteristics" in f]
                md_paths_subfolder = [md_paths_subfolder[0]] * len(image_paths_subfolder)

                self.image_paths.extend(image_paths_subfolder)
                self.md_paths.extend(md_paths_subfolder)
        
        #print(len(self.image_paths))
        #print(len(self.md_paths))
        self.transform = transform

        # Create all possible pairs of indices
        # self.pairs = list(itertools.combinations(range(len(self.image_paths)), 2))
        self.pairs = [tuple(random.sample(range(len(self.image_paths)), 2)) for _ in range(len(self.image_paths))]

        #self.all_metadata = metadata_normalization(self.md_paths)

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

        "NOTE: Think about using pytorch transform for normalizing images"
        # Normalize images between 0 and 1
        # Using min-max norm
        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

        return img1, img2
    
    def calculate_similarity(self, md1, md2, distance_function):
        truth = similarity_function(md1, md2, distance_function)
        return truth

    def __getitem__(self, index):
        img1, img2, md1, md2 = self.load_data(index)

        truth_label = self.calculate_similarity(md1, md2, self.distance_function)

        img1, img2 = self.normalize_images(img1, img2)

        img1 = np.array(img1)
        img2 = np.array(img2)
        truth_label = np.array(truth_label)
        #print("truth_label", truth_label)
        
        if self.transform:
            img1 = self.transform(img1) #NOTE: Make sure the transform converts to pytorch tensors
            img2 = self.transform(img2)


        return img1, img2, truth_label

    

def load_data(data_folder: str, 
              data_type="pkl", 
              train_size=0.6, 
              transform=None, 
              random_seed=None,
              distance_function="cosine"):
    
    """
    This function creates dataset using the SetupData class.

    Inputs: 
        data_folder - 

        data_type - 

        train_size - Set the fraction of data used for training.
            if train_size < 0.5:
                validation_size, test_size = train_size/2
                # Uses train_size*2 of data
                
            if train_size > 0.5: 
                validation_size, test_size = (1-train_size)/2
                # Uses all of data

            Set train_size=0.01 for testing the setup. 

    """
    
    if data_type == "pkl":
        custom_dataset = SetupData(data_folder=data_folder, 
                               transform=transform,
                               fraction=train_size,
                               distance_function=distance_function
                               )
    #else:
    #    custom_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    if random_seed is not None:
        np.random.seed(random_seed)
    
    val_size = 0.5 # 0.5 = Split validation and test sets evenly
    if train_size == 0.01: # For testing set train size = 0.01 and test_size will follow
        test_size = train_size
    elif train_size < 0.5:
        test_size = train_size # Make sure test is not larger than train
    else:
        test_size = 1-train_size # Use all data
    
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
    data_folder: str,
    dataloader_dir: str,
    data_type: str,
    train_size: float,
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int,
    random_seed: int,
    distance_function: str,
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
    train_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_train_{input_size}_{distance_function}.pth")
    validation_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_validation_{input_size}_{distance_function}.pth")
    test_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_test_{input_size}_{distance_function}.pth")
 
    if os.path.exists(train_dataloader_path) and os.path.exists(validation_dataloader_path):
        
        # If saved dataloaders exist, load them
        print("\nLoading dataloaders...")
        print(f"with train_size={train_size} and batch_size={batch_size}")

        train_dataloader = torch.load(train_dataloader_path)
        validation_dataloader = torch.load(validation_dataloader_path)
        test_dataloader = torch.load(test_dataloader_path)

        print(f"\n[INFO] Loaded saved dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation batches with batch_size={batch_size}")
        print(f"This gives a total of {len(train_dataloader.dataset)} training images and {len(validation_dataloader.dataset)} validation images")
    else:
        # Create new dataset (takes a while)
        print("Creating dataset...")
        print(f"with train_size={train_size} and batch_size={batch_size}")
        train_data, validation_data, test_data = \
            load_data(data_folder, 
                        data_type, 
                        train_size, 
                        transform, 
                        random_seed,
                        distance_function
                        )
        
        print(f"\nDataset created with {len(train_data)+len(validation_data)+len(test_data)} images with shape {train_data[0][0].shape}")

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False, # Set False to maintain the order of image pairs!
            num_workers=num_workers,
            pin_memory=True, # Set pin_memory to True for GPU acceleration
        )
        validation_dataloader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"\n[INFO] Created dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation and test batches with batch_size={batch_size}")
        print(f"This gives a total of {len(train_data)} training images and {len(validation_data)} validation and test images")
        print("Saving dataloaders...")
        # Check if the folder exists, create it if it doesn't
        if not os.path.exists(dataloader_dir):
            os.makedirs(dataloader_dir)

        # Save train and validation dataloaders
        torch.save(train_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_train_{input_size}_{distance_function}.pth"))
        torch.save(validation_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_validation_{input_size}_{distance_function}.pth"))
        torch.save(test_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_test_{input_size}_{distance_function}.pth"))
        print("Dataloaders saved at", dataloader_dir)

    return train_dataloader, validation_dataloader, test_dataloader