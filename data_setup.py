
import torch
#import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

import pickle
import os
import tqdm as tqdm
import itertools
import pandas as pd
#from sklearn import metrics
import random
#import math
import h5py
import time

"""
This script is for creating PyTorch dataloaders. 

NOTE: Dataloaders are saved and will not be over-written. Any changes made in this document 
will not be applied to existing dataloaders. When making changes, consider deleting existing dataloaders
to be certain that all changes are applied.

"""
save_dataset = False
shuffle=True, # whether or not to shuffle batches between each epoch

device_nr = "2"    
TRAIN_SIZE = 0.01
BATCH_SIZE = 4
NUM_WORKERS = 1
NUM_EPOCHS = 10
num_samples = 10
MARGIN = 0.9
PATIENCE = 5
MIN_DELTA = 1e-3
LR = 1e-4
L1 = 32
L2 = 32
L3 = 128
L4 = 64
L5 = 128
FC_UNITS1 = 512
FC_UNITS2 = 128
FC_UNITS3 = 1028
FC_UNITS4 = 4096
FC_UNITS5 = 128
LEARNING_RATE = 1e-4
targets = -1
RANDOM_SEED = None
output_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs"
model_name = "just_testing"
data_folder = "/lustre/astro/antonmol/atm_new_dataset_sink49"
dataloader_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders"
normalizer_file = "metadata_stats_sink49.csv"
normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/{normalizer_file}", index_col=0)



input_size = 512
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.CenterCrop(input_size), # Crop image to 400x400 pixels 
    #transforms.RandomRotation(45) # Randomly rotates between min/max of specified value
    # NOTE: Remember to change model_builder.input_size accordingly, if changing size of image by cropping

    # transforms.RandomErasing(),
    # Add transform here
])

md_names = ["ar_p", "ar_s", "mih", "m", "ds", "sep", "t"]



test_similarity = None

distance_function = "cosine"  # cosine, euclid
norm_type = "z_score" # minmax, z_score

##### -------------- Step 1: Create/load dataset ----------------


# min, max, mean and std values computed for each metadata parameter
ar_p_stats = [1.274263e-10 , 1.525225e-05 , 2.663110e-06, 2.644608e-06] #[min, max, mean, std]
ar_s_stats = [1.999744e-14 , 6.386982e-13 , 1.702182e-13 , 7.274838e-14]
mih_stats = [1.1012342557696856e+30 , 1.175988350929672e+32 , 2.262783690105513e+31 , 2.440384162054931e+31]
m_stats = [4.747706220749683e-05 , 0.0006432782806819867 , 0.000423637317739924 , 0.00015939441534300718]
ds_stats = [0.0 , 458.1103546350497 , 150.011442111849 , 94.89227491504303]
sep_stats = [225.1923007248629 , 5779.502318739806 , 1392.750789695731 , 1619.5235585032883]
t_stats = [4.15463518978504 , 166.8422452216837 , 85.73724726640633 , 48.529424497343555]


# -------------- Metadata extraction ----------------

def metadata_normalization(data_folders, name, data_type="pkl", save=True):
    
    """
    Calculate minimum and maximum for each parameter to normalize it. 
    Function not used in training loop
    """
    if not isinstance(data_folders, list):
            data_folders = [data_folders]

    print("Normalizing metadata...")
    metadata_files = []

    if data_type == "pkl":
        
        ar_p = []
        ar_s = []
        mih = []
        m = []
        ds = []
        sep = []
        t = []

        for data_folder in data_folders:
            for subfolder in os.listdir(data_folder):
                subfolder_path = os.path.join(data_folder, subfolder)
                md_file = [os.path.join(data_folder, subfolder, f) for f in os.listdir(subfolder_path) if "characteristics" in f][0]
                with open(md_file, 'rb') as f:
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
                
        print(f"Gathered all {len(ar_p)} params")
        all_metadata = [ar_p, ar_s, mih, m, ds, sep, t]
        all_metadata_names = ['ar_p_stats', 'ar_s_stats', 'mih_stats', 'm_stats', 'ds_stats', 'sep_stats', 't_stats']

    elif data_type == "krisha":
        m_env, ar, temp, m, ds, lum, t = [], [], [], [], [], [], []

        for data_folder in data_folders:
            for file in os.listdir(data_folder):
                md_file = os.path.join(data_folder, file)
                try:

                    with h5py.File(md_file, "r") as hf:
                        m_env.append(hf['Envelope_mass'][()])
                        ar.append(hf['accretion_rate'][()])
                        temp.append(hf['temperature'][()])
                        m.append(hf['mass'][()])
                        ds.append(hf['disk_size'][()])
                        lum.append(hf['luminosity'][()])
                        t.append(hf['age'][()])

                except Exception as e:
                    print(e)
                    print(f"Error for file {md_file}")
        all_metadata = [m_env, ar, temp, m, ds, lum, t]
        all_metadata_names = ["Envelope_mass", "accretion_rate", "temperature", "mass", "disk_size", "luminosity", "age"]

    md_stats = {}
    for i, metadata in enumerate(all_metadata):
        md_stats[all_metadata_names[i]] = [np.min(metadata), np.max(metadata), np.mean(metadata), np.std(metadata)]
    
    md_stats_df = pd.DataFrame(md_stats, index=['min', 'max', 'mean', 'std'])
    #md_stats_df = md_stats_df.T
    if save == True:
        save_file_df = f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_{name}.csv"
        md_stats_df.to_csv(save_file_df)
        print(f"Saved metadata statistics at {save_file_df}")
    
    return md_stats_df, all_metadata, all_metadata_names


def get_metadata(md, md_names, normalizer, filetype="dataframe", norm_type="z_score", data_type="pkl"):

    if data_type == "pkl":
        S = md
        p = md["primary sink tag"]

        md_p_raw = {
            "ar_p": S['accretion_rate_primary'], 
            "ar_s": S['accretion_rate_secondary'].value, 
            "mih": S['mass_in_100'],
            "m": S["rsink file"]["m"][p], 
            "ds": S['disk size'], 
            "sep": S['separation'].value,
            "t": S['t_after_formation'].value, 
                                    }

    elif data_type == "krisha":
        with h5py.File(md, "r") as hf:

            md_p_raw = {
                "m_env": hf['Envelope_mass'][()], 
                "ar":    hf['accretion_rate'][()], 
                "temp":  hf['temperature'][()],
                "m":     hf["mass"][()], 
                "ds":    hf['disk_size'][()], 
                "lum":   hf['luminosity'][()],
                "t":     hf['age'][()], 
                                        }

            
    if norm_type == "minmax":
        if data_type == "pkl":
            md_p = {"ar_p": (S['accretion_rate_primary'] - normalizer["ar_p_stats"]["min"]) / 
                    (normalizer["ar_p_stats"]["max"] - normalizer["ar_p_stats"]["min"]), 
                    "ar_s": ((S['accretion_rate_secondary'].value) - normalizer["ar_s_stats"]["min"]) / 
                    (normalizer["ar_s_stats"]["max"] - normalizer["ar_s_stats"]["min"]), 
                    "mih": ((S['mass_in_100']) - normalizer["mih_stats"]["min"]) / 
                    (normalizer["mih_stats"]["max"] - normalizer["mih_stats"]["min"]),
                    "m": ((S["rsink file"]["m"][p]) - normalizer["m_stats"]["min"]) / 
                    (normalizer["m_stats"]["max"] - normalizer["m_stats"]["min"]), 
                    "ds": ((S['disk size']) - normalizer["ds_stats"]["min"]) / 
                    (normalizer["ds_stats"]["max"] - normalizer["ds_stats"]["min"]), 
                    "sep": ((S['separation'].value) - normalizer["sep_stats"]["min"]) / 
                    (normalizer["sep_stats"]["max"] - normalizer["sep_stats"]["min"]),
                    "t": ((S['t_after_formation'].value) - normalizer["t_stats"]["min"]) / 
                    (normalizer["t_stats"]["max"] - normalizer["t_stats"]["min"]), 
                                    }
        elif data_type == "krisha":
            with h5py.File(md, "r") as hf:
                md_p = {
                        "m_env": (hf['Envelope_mass'][()] - normalizer["Envelope_mass"]["min"]) / 
                                (normalizer["Envelope_mass"]["max"] - normalizer["Envelope_mass"]["min"]), 
                        "ar": (hf['accretion_rate'][()] - normalizer["accretion_rate"]["min"]) / 
                                (normalizer["accretion_rate"]["max"] - normalizer["accretion_rate"]["min"]), 
                        "temp": (hf['temperature'][()] - normalizer["temperature"]["min"]) / 
                                (normalizer["temperature"]["max"] - normalizer["temperature"]["min"]),
                        "m": (hf["mass"][()] - normalizer["mass"]["min"]) / 
                                (normalizer["mass"]["max"] - normalizer["mass"]["min"]), 
                        "ds": (hf['disk_size'][()] - normalizer["disk_size"]["min"]) / 
                                (normalizer["disk_size"]["max"] - normalizer["disk_size"]["min"]), 
                        "lum": (hf['luminosity'][()] - normalizer["luminosity"]["min"]) / 
                                (normalizer["luminosity"]["max"] - normalizer["luminosity"]["min"]),
                        "t": (hf['age'][()] - normalizer["age"]["min"]) / 
                                (normalizer["age"]["max"] - normalizer["age"]["min"]), 
                    }
                    
  

    elif norm_type == "z_score":
        if data_type == "pkl":
            md_p = {
                    "ar_p": (S["accretion_rate_primary"] - normalizer["ar_p_stats"]["mean"]) / (normalizer["ar_p_stats"]["std"]), 
                    "ar_s": (S["accretion_rate_secondary"].value - normalizer["ar_s_stats"]["mean"]) / (normalizer["ar_s_stats"]["std"]),
                    "mih" : (S["mass_in_100"] - normalizer["mih_stats"]["mean"]) / (normalizer["mih_stats"]["std"]),  
                    "m": (S["rsink file"]["m"][p] - normalizer["m_stats"]["mean"]) / (normalizer["m_stats"]["std"]), 
                    "ds": (S["disk size"] - normalizer["ds_stats"]["mean"]) / (normalizer["ds_stats"]["std"]),
                    "sep": (S["separation"].value - normalizer["sep_stats"]["mean"]) / (normalizer["sep_stats"]["mean"]), 
                    "t": (S["t_after_formation"].value - normalizer["t_stats"]["mean"]) / (normalizer["t_stats"]["std"]),
                        }
        elif data_type == "krisha":
            with h5py.File(md, "r") as hf:
 
                md_p = {
                        "m_env": (hf["Envelope_mass"][()] - normalizer["Envelope_mass"]["mean"]) / normalizer["Envelope_mass"]["std"], 
                        "ar":    (hf['accretion_rate'][()] - normalizer["accretion_rate"]["mean"]) / normalizer["accretion_rate"]["std"], 
                        "temp":  (hf['temperature'][()] - normalizer["temperature"]["mean"]) / normalizer["temperature"]["std"],
                        "m":     (hf["mass"][()] - normalizer["mass"]["mean"]) / normalizer["mass"]["std"], 
                        "ds":    (hf['disk_size'][()] - normalizer["disk_size"]["mean"]) / normalizer["disk_size"]["std"], 
                        "lum":   (hf['luminosity'][()] - normalizer["luminosity"]["mean"]) / normalizer["luminosity"]["std"],
                        "t":     (hf['age'][()] - normalizer["age"]["mean"]) / normalizer["age"]["std"], 
                    }
                

 
    elif norm_type == None:
        md_p = md_p_raw
    

    if filetype == "dataframe":
        primary_md_all = pd.DataFrame(md_p, index=["Primary"])
        primary_md_raw_all = pd.DataFrame(md_p_raw, index=["Primary_raw"])

        primary_md = primary_md_all[md_names] # Only use those parameters defined in md_names
        primary_md_raw = primary_md_raw_all[md_names]

        #secondary_md = pd.DataFrame(md_s, index=["Secondary"]) 
        #metadata = pd.concat([primary_md, secondary_md])
  
    return primary_md, primary_md_raw

# -------------- Similarity function --------------------
def similarity_function(md1, md2, md_names, normalizer, distance_function="cosine", data_type="pkl"):

    MD1, MD1_raw = get_metadata(md1, md_names, normalizer, filetype="dataframe", norm_type=norm_type, data_type=data_type)
    MD2, MD2_raw = get_metadata(md2, md_names, normalizer, filetype="dataframe", norm_type=norm_type, data_type=data_type)
    
    if data_type == "pkl":
        MD1_array = np.array(MD1)[0][0]
        MD2_array = np.array(MD2)[0][0]
    elif data_type == "krisha":
        MD1_array = np.array(MD1)[0]
        MD2_array = np.array(MD2)[0]

    MD1_tensor = torch.tensor(MD1_array)  # Convert MD1 to a PyTorch tensor
    MD2_tensor = torch.tensor(MD2_array)

    # print("md1", MD1_tensor)
    # print("md2", MD2_tensor)
    if distance_function == "cosine":
        truth = F.cosine_similarity(MD1_tensor, MD2_tensor, dim=0)
        
        # truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
        # truth = 1-truth
        # x = 5 * (truth - 0.5)
        # truth = 1 / (1 + np.exp(-x))
        
        #print("truthshape", truth.shape)
        #print("TRUTH", truth)

        truth = truth.ravel()[0]
        #print("TRUTH", truth)

    if distance_function == "euclid":
        #truth = metrics.euclidean_distances(MD1_log, MD2_log)
        #truth = np.sqrt(np.sum((MD1 - MD2)**2))
        truth = F.pairwise_distance(MD1_tensor, MD2_tensor)
        #truth = torch.sigmoid(-truth_tensor)
        #truth = truth_tensor.numpy()
        #truth = truth.ravel()[0]
        
    #print("TRUTH", truth)
        
    return truth, MD1, MD2, MD1_raw, MD2_raw



# class SetupData(Dataset):

#     """
#     This class is used to get all images within a target directory.


    
#     """

#     def __init__(self, data_folder, md_names, normalizer, transform=None, fraction=0.01, distance_function="cosine", data_type="pkl",
#                  specific_image_paths=None, specific_md_paths=None,
#                  ):
#         self.image_folder = data_folder
#         # self.image_paths = []
#         # self.md_paths = []
#         self.image_paths = specific_image_paths if specific_image_paths is not None else []
#         self.md_paths = specific_md_paths if specific_md_paths is not None else []
#         self.distance_function = distance_function
#         self.md_names = md_names
#         self.normalizer = normalizer
#         self.counter = 0
#         self.data_type = data_type
#         self.transform = transform
        
#         if not specific_image_paths and not specific_md_paths:
#             # Iterate over subfolders within data_folder
#             if self.data_type == "pkl":
#                 all_subfolders = [subfolder for subfolder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, subfolder))]
#                 size = int(1 * len(all_subfolders))
#                 #print(f"Getting {size} image pairs")
#                 selected_subfolders = random.sample(all_subfolders, size)
#                 for subfolder in selected_subfolders:
#                     subfolder_path = os.path.join(data_folder, subfolder)

#                     if os.path.isdir(subfolder_path):
#                         image_paths_subfolder = [os.path.join(data_folder, subfolder_path, f) for f in os.listdir(subfolder_path) if "projection" in f]
#                         self.image_paths.extend(image_paths_subfolder)
                        
#                         md_paths_subfolder = [os.path.join(data_folder, subfolder, f) for f in os.listdir(subfolder_path) if "characteristics" in f]
#                         md_paths_subfolder = [md_paths_subfolder[0]] * len(image_paths_subfolder)
#                         self.md_paths.extend(md_paths_subfolder)


#             elif self.data_type == "krisha":
#                 if not isinstance(data_folder, list):
#                     data_folder = [data_folder]
                
#                 if len(data_folder) > 1:
#                     print(f"Using data for {len(data_folder)} sinks:")
#                     print(data_folder)
#                 # img_files = []
#                 # for folder in data_folder:
#                 #     files_in_folder = os.listdir(folder)
#                 #     img_files.extend(files_in_folder)
#                 # img_files = os.listdir(data_folder)
#                 # size = int(1 * len(img_files))
#                 # selected_images = random.sample(img_files, size)
#                 # for img_file in selected_images:
#                 #     image_path = os.path.join(data_folder, img_file)
#                 #     self.image_paths.append(image_path)
#                 for folder in data_folder:
#                     for root, dirs, files in os.walk(folder):
#                         for file in files:
#                             file_path = os.path.join(root, file)
#                             self.image_paths.append(file_path)

            

#         # Create all possible pairs of indices
#         # self.pairs = list(itertools.combinations(range(len(self.image_paths)), 2))
#         print(f"There are {len(self.image_paths)} images")
#         total_combinations = (len(self.image_paths) * (len(self.image_paths) - 1)) / 2
#         print(f"...which gives {total_combinations} possible combinations.")

#         if data_type == "pkl":
#             step_size = int(total_combinations / (fraction*len(self.image_paths)))
#             self.pairs = list(itertools.combinations(range(len(self.image_paths)), 2))[::step_size]
#             print(f"Taking only every combs/({fraction}*imgs)={step_size}th combination gives {len(self.pairs)} pairs")
        
#         elif data_type == "krisha":
#             combinations = list(itertools.combinations(range(len(self.image_paths)), 2))
#             random.shuffle(combinations)
#             num_combinations = len(combinations)
#             num_pairs = int(num_combinations * fraction)
#             self.pairs = combinations[:num_pairs]
#             print(f"Selected {num_pairs} random pairs out of {num_combinations}")
#             # step_size = int((1-fraction)*100)
#             # self.pairs = list(itertools.combinations(range(len(self.image_paths)), 2))[::step_size]
#             # print(f"Taking every (1-{fraction})*100 = {step_size}th combination gives {len(self.pairs)} pairs")
#         self.tenth_of_total = len(self.pairs) // 10
#         #self.pairs = [tuple(random.sample(range(len(self.image_paths)), 2)) for _ in range(len(self.image_paths))]
        
#         #self.all_metadata = metadata_normalization(self.md_paths)

#     def load_data(self, index: int):
#         "Opens an image and its metadata via a path and returns them."
#         index1, index2 = self.pairs[index]
        
#         if self.data_type == "pkl":
#             # Load image data
#             with open(self.image_paths[index1], 'rb') as f:
#                 img1 = pickle.load(f)
#             with open(self.image_paths[index2], 'rb') as f:
#                 img2 = pickle.load(f)
                
#             # Load metadata
#             with open(self.md_paths[index1], 'rb') as f:
#                 md1 = pickle.load(f)
#             with open(self.md_paths[index2], 'rb') as f:
#                 md2 = pickle.load(f)
        
#         elif self.data_type == "krisha":
            
#             with h5py.File(self.image_paths[index1], 'r') as f:
#                 rand_img = np.random.randint(1, 9) #get random projection
#                 img1 = f[f'intensity_projection_{rand_img}'][:]
#                 md1 = self.image_paths[index1]
#             with h5py.File(self.image_paths[index2], 'r') as f:
#                 rand_img = np.random.randint(1, 9) #get random projection
#                 img2 = f[f'intensity_projection_{rand_img}'][:]
#                 md2 = self.image_paths[index2]

#         self.counter += 1

#         if self.counter == self.tenth_of_total:
#             self.start_time = time.time()
#         # Print progress at 10% intervals
#         if self.counter % self.tenth_of_total == 0:
#             print(f"Created {self.counter} out of {len(self.pairs)} image pairs")
#             if self.counter == self.tenth_of_total * 2:
#                 end_time = time.time()
#                 elapsed_time = end_time - self.start_time
#                 print(f"It took {elapsed_time:.1f}s to create 10% of data.")


#         return img1, img2, md1, md2

#     def __len__(self):
#         return len(self.pairs)
        
#     def normalize_images(self, img1, img2):

#         "NOTE: Think about using pytorch transform for normalizing images"
#         # Normalize images between 0 and 1
#         # Using min-max norm
#         img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
#         img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

#         return img1, img2
    
#     def calculate_similarity(self, md1, md2, md_names, normalizer, distance_function, data_type="pkl"):
#         truth, MD1, MD2, MD1_raw, MD2_raw = similarity_function(md1, md2, md_names, normalizer, distance_function, data_type=data_type)
#         return truth, MD1, MD2, MD1_raw, MD2_raw

#     def __getitem__(self, index):
#         img1, img2, md1, md2 = self.load_data(index)

#         truth_label, MD1, MD2, MD1_raw, MD2_raw = self.calculate_similarity(md1, md2, self.md_names, self.normalizer, self.distance_function, self.data_type)

#         img1, img2 = self.normalize_images(img1, img2)

#         img1 = np.array(img1)
#         img2 = np.array(img2)
#         truth_label = np.array(truth_label)
#         #print("truth_label", truth_label)
        
#         if self.transform:
#             img1 = self.transform(img1) #NOTE: Make sure the transform converts to pytorch tensors
#             img2 = self.transform(img2)

#         # # Get metadata
#         # MD1, MD1_raw = get_metadata(md1, self.md_names, self.normalizer, filetype="dataframe", norm_type=norm_type, data_type="pkl")
#         # MD2, MD2_raw = get_metadata(md2, self.md_names, self.normalizer, filetype="dataframe", norm_type=norm_type, data_type="pkl")

#         return img1, img2, truth_label, np.array(MD1), np.array(MD2), np.array(MD1_raw), np.array(MD2_raw)


class SetupData(Dataset):
    """
    This class is used to get all images within a target directory.
    """

    def __init__(self, data_folder, md_names, normalizer, transform=None, fraction=0.01, distance_function="cosine", data_type="pkl", specific_image_paths=None, specific_md_paths=None):
        self.image_folder = data_folder
        self.image_paths = specific_image_paths if specific_image_paths is not None else []
        self.md_paths = specific_md_paths if specific_md_paths is not None else []
        self.distance_function = distance_function
        self.md_names = md_names
        self.normalizer = normalizer
        self.counter = 0
        self.data_type = data_type
        self.transform = transform

        

        if not self.image_paths:
            print("USING ALL DATA")
            if not isinstance(data_folder, list):
                data_folder = [data_folder]
            if self.data_type == "krisha":
                for folder in data_folder:
                    for root, dirs, files in os.walk(folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            self.image_paths.append(file_path)
            else:
                raise ValueError("Unsupported data type or missing image paths.")
            
            if not self.image_paths:
                raise ValueError("No image paths found.")

        # Create all possible pairs of indices
        total_combinations = (len(self.image_paths) * (len(self.image_paths) - 1)) // 2
        print(f"There are {len(self.image_paths)} images, which gives {total_combinations} possible combinations.")

        #step_size = max(1, int(total_combinations / (fraction * len(self.image_paths))))
        step_size = max(1, int(total_combinations / (fraction * total_combinations)))
        self.pairs = list(itertools.combinations(range(len(self.image_paths)), 2))[::step_size]
        print(f"Taking every {step_size}-th combination gives {len(self.pairs)} pairs.")

        self.tenth_of_total = max(1, len(self.pairs) // 10)

    def load_data(self, index: int):
        "Opens an image and its metadata via a path and returns them."
        index1, index2 = self.pairs[index]
        
        if self.data_type == "krisha":
            with h5py.File(self.image_paths[index1], 'r') as f:
                rand_img = np.random.randint(1, 9) # Get random projection
                img1 = f[f'intensity_projection_{rand_img}'][:]
                md1 = self.image_paths[index1]
            with h5py.File(self.image_paths[index2], 'r') as f:
                rand_img = np.random.randint(1, 9) # Get random projection
                img2 = f[f'intensity_projection_{rand_img}'][:]
                md2 = self.image_paths[index2]

        self.counter += 1
        if self.counter % self.tenth_of_total == 0:
            print(f"Created {self.counter} out of {len(self.pairs)} image pairs")

        return img1, img2, md1, md2

    def __len__(self):
        return len(self.pairs)

    def normalize_images(self, img1, img2):
        "Normalize images between 0 and 1 using min-max normalization."
        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
        return img1, img2
    
    def calculate_similarity(self, md1, md2, md_names, normalizer, distance_function, data_type="pkl"):
        truth, MD1, MD2, MD1_raw, MD2_raw = similarity_function(md1, md2, md_names, normalizer, distance_function, data_type=data_type)
        return truth, MD1, MD2, MD1_raw, MD2_raw

    def __getitem__(self, index):
        img1, img2, md1, md2 = self.load_data(index)
        truth_label, MD1, MD2, MD1_raw, MD2_raw = self.calculate_similarity(md1, md2, self.md_names, self.normalizer, self.distance_function, self.data_type)
        img1, img2 = self.normalize_images(img1, img2)

        img1 = np.array(img1)
        img2 = np.array(img2)
        truth_label = np.array(truth_label)

        if self.transform:
            img1 = self.transform(img1) # Make sure the transform converts to pytorch tensors
            img2 = self.transform(img2)

        return img1, img2, truth_label, np.array(MD1), np.array(MD2), np.array(MD1_raw), np.array(MD2_raw)






# def load_data(data_folder: str, 
#               md_names,
#               normalizer,
#               data_type="pkl", 
#               train_size=0.6, 
#               transform=None, 
#               random_seed=None,
#               distance_function="cosine",
#               ):
    
#     """
#     This function creates dataset using the SetupData class.

#     Inputs: 
#         data_folder - 

#         data_type - 

#         train_size - Set the fraction of data used for training.
#             if train_size < 0.5:
#                 validation_size, test_size = train_size/2
#                 # Uses train_size*2 of data
                
#             if train_size > 0.5: 
#                 validation_size, test_size = (1-train_size)/2
#                 # Uses all of data

#             Set train_size=0.01 for testing the setup. 

#     """
    
#     if data_type == "pkl" or data_type == "krisha":
#         custom_dataset = SetupData(data_folder=data_folder, 
#                                    md_names=md_names,
#                                    normalizer=normalizer,
#                                    transform=transform,
#                                    fraction=train_size,
#                                    distance_function=distance_function,
#                                    data_type=data_type
#                                )
#     #else:
#     #    custom_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


#     if random_seed is not None:
#         np.random.seed(random_seed)
    
#     # if train_size == 0.01: # For testing set train size = 0.01 and test_size will follow
#     #     test_size = train_size
#     # elif train_size < 0.5:
#     #     test_size = train_size # Make sure test is not larger than train
#     # else:
#     #     test_size = 1-train_size # Use all data
#     trying_random = False

#     # print("Custom dataset before:")
#     # print(len(custom_dataset))
#     # print(custom_dataset)

#     # print(len(custom_dataset[0 : int(0.5*len(custom_dataset))]))
#     # print(custom_dataset[0 : int(0.5*len(custom_dataset))])

#     if trying_random == True:
#         train_dataset = custom_dataset[0 : int(0.5*len(custom_dataset))]
#         val_dataset = custom_dataset[int(0.5*len(custom_dataset)) : int(0.75*len(custom_dataset))]
#         test_dataset = custom_dataset[int(0.75*len(custom_dataset)) : ]
#     else:
#         train_size = 0.5
#         test_size = 0.5
        
#         # Split data into training and the rest
#         train_dataset, temp_dataset = train_test_split(custom_dataset, 
#                                                         train_size=train_size, 
#                                                         test_size=test_size,
#                                                         random_state=random_seed,
#                                                         shuffle=False
#         )
#         # Split the remainder to equal sized validation and testing sets
#         val_size = 0.5 # 0.5 = Split validation and test sets evenly
#         val_dataset, test_dataset = train_test_split(temp_dataset, 
#                                                         train_size=val_size, 
#                                                         test_size=val_size,
#                                                         random_state=random_seed,
#                                                         shuffle=False
#         )
    
#     return train_dataset, val_dataset, test_dataset



def load_data(data_folder, 
              md_names,
              normalizer,
              data_type="pkl", 
              train_size=0.6, 
              transform=None, 
              random_seed=None,
              distance_function="cosine"):
    """
    This function creates datasets using the SetupData class.

    Inputs: 
        data_folder - Directory containing the data.
        data_type - Type of data ("pkl" or "krisha").
        train_size - Fraction of data used for training.
            If train_size < 0.5, validation_size and test_size = train_size / 2.
            If train_size > 0.5, validation_size and test_size = (1 - train_size) / 2.
        transform - Transformations to apply to the data.
        random_seed - Seed for random number generation.
    """
    if not isinstance(data_folder, list):
        data_folder = [data_folder]

    # Collect all image paths
    if data_type == "krisha":
        image_paths = []
        for folder in data_folder:
            for root, dirs, files in os.walk(folder):
                #print(files)
                for file in files:
                    file_path = os.path.join(root, file)
                    image_paths.append(file_path)
        
        if not image_paths:
            raise ValueError("No image paths found.")

        # Split data paths
        train_paths, temp_paths = train_test_split(image_paths, 
                                                   train_size=train_size, 
                                                   random_state=random_seed, 
                                                   shuffle=True)
        val_size = train_size / 2
        val_paths, test_paths = train_test_split(temp_paths, 
                                                 train_size=val_size, 
                                                 test_size=val_size,
                                                 random_state=random_seed, 
                                                 shuffle=True)

        # Create datasets
        train_dataset = SetupData(data_folder=data_folder, 
                                  md_names=md_names,
                                  normalizer=normalizer,
                                  transform=transform,
                                  fraction=0.1,
                                  distance_function=distance_function,
                                  data_type=data_type,
                                  specific_image_paths=train_paths)
        
        train_dataset, _ = train_test_split(train_dataset, 
                                            train_size=0.99, 
                                            random_state=None, 
                                            shuffle=False)

        val_dataset = SetupData(data_folder=data_folder, 
                                md_names=md_names,
                                normalizer=normalizer,
                                transform=transform,
                                fraction=0.2,
                                distance_function=distance_function,
                                data_type=data_type,
                                specific_image_paths=val_paths)
        
        val_dataset, _ = train_test_split(val_dataset, 
                                            train_size=0.99, 
                                            random_state=None, 
                                            shuffle=False)

        test_dataset = SetupData(data_folder=data_folder, 
                                 md_names=md_names,
                                 normalizer=normalizer,
                                 transform=transform,
                                 fraction=0.2,
                                 distance_function=distance_function,
                                 data_type=data_type,
                                 specific_image_paths=test_paths)
        
        test_dataset, _ = train_test_split(test_dataset, 
                                            train_size=0.99, 
                                            random_state=None, 
                                            shuffle=False)
        

    else:
        raise ValueError("Only 'krisha' data type is currently supported.")

    return train_dataset, val_dataset, test_dataset


def save_datasets_hdf5(datasets, filename):
    with h5py.File(filename, 'w') as f:
        for name, dataset in datasets.items():
            f.create_dataset(name, data=dataset)

# Function to load datasets
def load_datasets_hdf5(filename):
    datasets = {}
    with h5py.File(filename, 'r') as f:
        for name in f.keys():
            datasets[name] = f[name][:]
    return datasets











# -------------- Create dataloader -----------------------


# def create_dataloaders(
#     data_folder,
#     md_names,
#     normalizer,
#     dataloader_dir: str,
#     data_type: str,
#     train_size: float,
#     transform: transforms.Compose, 
#     batch_size: int, 
#     num_workers: int,
#     random_seed: int,
#     distance_function: str,
#     ):

#     """Creates training and validation DataLoaders.

#     Takes in a data path and turns it into training and validation PyTorch Datasets 
#     and then into PyTorch DataLoaders. The fraction of data used for training 
#     is specified with train_size between 0 and 1 with 1 being all data used for training.

#     Args:
#     data_dir: Path to data directory.
#     data_type: type of data (png, jpg, pkl, ...)
#     transform: torchvision transforms to perform on training and validation data.
#     batch_size: Number of samples per batch in each of the DataLoaders.
#     num_workers: An integer for number of workers per DataLoader.

#     Returns:
#     A tuple of (train_dataloader, validation_dataloader).

#     Example usage:
#         train_dataloader, validation_dataloader = \
#         = create_dataloaders(data_dir=path/to/data_dir,
#                                 md_names=md_names
#                                 data_type='pkl',
#                                 train_size=0.8,
#                                 transform=some_transform,
#                                 batch_size=32,
#                                 num_workers=4)
#     """
    
#     # Check if saved dataloaders exist
#     # train_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_train_{distance_function}_md_{md_names}.pth")
#     # validation_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_validation_{distance_function}_md_{md_names}.pth")
#     # test_dataloader_path = os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_test_{distance_function}_md_{md_names}.pth")
#     if data_type == "pkl":
#         dataset_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/datasets"
#     elif data_type == "krisha":
#         dataset_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/datasets_krisha"
#     train_dataset_path = os.path.join(dataset_dir, f"tsize{train_size}_train_{distance_function}_md_{md_names}_datasets.pth")
#     validation_dataset_path = os.path.join(dataset_dir, f"tsize{train_size}_validation_{distance_function}_md_{md_names}_datasets.pth")
#     test_dataset_path = os.path.join(dataset_dir, f"tsize{train_size}_test_{distance_function}_md_{md_names}_datasets.pth")
    
#     # if os.path.exists(train_dataloader_path) and os.path.exists(validation_dataloader_path):
        
#     #     # If saved dataloaders exist, load them
#     #     print("\nLoading dataloaders...")
#     #     print(f"with train_size={train_size} and batch_size={batch_size}")

#     #     train_dataloader = torch.load(train_dataloader_path)
#     #     validation_dataloader = torch.load(validation_dataloader_path)
#     #     test_dataloader = torch.load(test_dataloader_path)

#     #     print(f"\n[INFO] Loaded saved dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation batches with batch_size={batch_size}")
#     #     print(f"This gives a total of {len(train_dataloader.dataset)} training images and {len(validation_dataloader.dataset)} validation images")
#     #     return train_dataloader, validation_dataloader, test_dataloader
    
#     if os.path.exists(train_dataset_path) and os.path.exists(validation_dataset_path): 
#         # Load datasets and create dataloaders
#         print("Loading dataset...")
#         print(f"from: {train_dataset_path}")
#         train_data = torch.load(train_dataset_path)
#         validation_data = torch.load(validation_dataset_path)
#         test_data = torch.load(test_dataset_path)

#     else:
#         # Create new dataset (takes a while)
#         print("Creating dataset...")
#         print(f"with train_size={train_size} and batch_size={batch_size}")
#         train_data, validation_data, test_data = \
#             load_data(data_folder, 
#                       md_names,
#                       normalizer,
#                       data_type, 
#                       train_size, 
#                       transform, 
#                       random_seed,
#                       distance_function,
#                         )
        
#         # if np.isnan(train_data).any() or np.isnan(validation_data).any() or np.isnan(test_data).any():
#         #     print("some of the data is nan")
#         print(f"\nDataset created with {len(train_data)+len(validation_data)+len(test_data)} images with shape {train_data[0][0].shape}")
        
#         if save_dataset == True:
#             print("Saving datasets...")
#             torch.save(train_data, os.path.join(dataset_dir, f'tsize{train_size}_train_{distance_function}_md_{md_names}_datasets.pth'))
#             torch.save(validation_data, os.path.join(dataset_dir, f'tsize{train_size}_validation_{distance_function}_md_{md_names}_datasets.pth'))
#             torch.save(test_data, os.path.join(dataset_dir, f'tsize{train_size}_test_{distance_function}_md_{md_names}_datasets.pth'))
#             print(f"Saved datasets at {dataset_dir}")
#             #print(f"Saved file:", f'tsize{train_size}_train_{distance_function}_md_{md_names}_datasets.pth')

#     # Turn datasets into data loaders
#     print("Creating dataloaders")
#     train_dataloader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=False, # Set False to maintain the order of image pairs! Is this necessary?
#         num_workers=num_workers,
#         pin_memory=True, # Set pin_memory to True for GPU acceleration
#     )
#     validation_dataloader = DataLoader(
#         validation_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#     test_dataloader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     print(f"\n[INFO] Created dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation and test batches with batch_size={batch_size}")
#     print(f"This gives a total of {len(train_data)} training images and {len(validation_data)} validation and test images")
#     # print("Saving dataloaders...")
#     # # Check if the folder exists, create it if it doesn't
#     # if not os.path.exists(dataloader_dir):
#     #     os.makedirs(dataloader_dir)

#     # # Save train and validation dataloaders
#     # torch.save(train_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_train_{distance_function}_md_{md_names}.pth"))
#     # torch.save(validation_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_validation_{distance_function}_md_{md_names}.pth"))
#     # torch.save(test_dataloader, os.path.join(dataloader_dir, f"tsize{train_size}_bsize{batch_size}_test_{distance_function}_md_{md_names}.pth"))
#     # print("Dataloaders saved at", dataloader_dir)

#     return train_dataloader, validation_dataloader, test_dataloader



def create_dataloaders(
    data_folder,
    md_names,
    normalizer,
    dataloader_dir: str,
    data_type: str,
    train_size: float,
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int,
    random_seed: int,
    distance_function: str,
    save_dataset=True,
):
    """Creates training, validation, and test DataLoaders."""
    
    dataset_dir = os.path.join(dataloader_dir, f"datasets_{data_type}")
    train_dataset_path = os.path.join(dataset_dir, f"tsize{train_size}_train_{distance_function}_md_{md_names}_datasets.pth")
    validation_dataset_path = os.path.join(dataset_dir, f"tsize{train_size}_validation_{distance_function}_md_{md_names}_datasets.pth")
    test_dataset_path = os.path.join(dataset_dir, f"tsize{train_size}_test_{distance_function}_md_{md_names}_datasets.pth")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if os.path.exists(train_dataset_path) and os.path.exists(validation_dataset_path) and os.path.exists(test_dataset_path):
        print("Loading datasets...")
        train_data = torch.load(train_dataset_path)
        validation_data = torch.load(validation_dataset_path)
        test_data = torch.load(test_dataset_path)
    else:
        print("Creating dataset...")
        print(f"with train_size={train_size} and batch_size={batch_size}")
        
        train_data, validation_data, test_data = \
            load_data(data_folder, 
                      md_names,
                      normalizer,
                      data_type, 
                      train_size, 
                      transform, 
                      random_seed,
                      distance_function,
                        )
        
        if save_dataset:
            print("Saving datasets...")
            torch.save(train_data, train_dataset_path)
            torch.save(validation_data, validation_dataset_path)
            torch.save(test_data, test_dataset_path)
            print(f"Saved datasets at {dataset_dir}")

    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n[INFO] Created dataloaders with {len(train_dataloader)} training batches and {len(validation_dataloader)} validation and test batches with batch_size={batch_size}")
    print(f"This gives a total of {len(train_data)} training images and {len(validation_data)} validation images")
    
    return train_dataloader, validation_dataloader, test_dataloader
