
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import tqdm as tqdm
import itertools
import pandas as pd
import random
import h5py
import time

"""
This script is for setting up PyTorch DataLoaders with pairs of images and a label for how 
similar the images are, based in their metadata.

"""


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
    if save == True:
        save_file_df = f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_{name}.csv"
        md_stats_df.to_csv(save_file_df)
        print(f"Saved metadata statistics at {save_file_df}")
    
    return md_stats_df, all_metadata, all_metadata_names


def get_metadata(md, md_names, normalizer, filetype="dataframe", norm_type="z_score", data_type="pkl"):
    """
    Collect and normalize metadata
    """
    
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

    if distance_function == "cosine":
        truth = F.cosine_similarity(MD1_tensor, MD2_tensor, dim=0)
        truth = truth.ravel()[0]

    if distance_function == "euclid":
        truth = F.pairwise_distance(MD1_tensor, MD2_tensor)
        
    return truth, MD1, MD2, MD1_raw, MD2_raw



class SetupData(Dataset):
    """
    This class is used to set up datasets of pairs of images, 
    along with their respective metadata for labelling.
    """

    def __init__(self, data_folder, md_names, normalizer, transform=None, fraction=0.01, distance_function="cosine", 
                 data_type="pkl", specific_image_paths=None, specific_md_paths=None):
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

        
        total_combinations = (len(self.image_paths) * (len(self.image_paths) - 1)) // 2 # create all possible pairs of indices
        print(f"There are {len(self.image_paths)} images, which gives {total_combinations} possible combinations.")

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
                rand_img = np.random.randint(1, 9)
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
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, truth_label, np.array(MD1), np.array(MD2), np.array(MD1_raw), np.array(MD2_raw)



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
    """
                  
    if not isinstance(data_folder, list):
        data_folder = [data_folder]

    if data_type == "krisha":
        image_paths = []
        for folder in data_folder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    image_paths.append(file_path)
        
        if not image_paths:
            raise ValueError("No image paths found.")

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

def load_datasets_hdf5(filename):
    datasets = {}
    with h5py.File(filename, 'r') as f:
        for name in f.keys():
            datasets[name] = f[name][:]
    return datasets



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
    """
    Creates training, validation, and test DataLoaders.
    """
    
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
