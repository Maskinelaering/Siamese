print("Loading modules...")

import train, data_setup, model_builder, utils
from torchvision import transforms
import pandas as pd
import traceback
import numpy as np
import itertools

import os


"This script is for running several experiments using the script from train.py"

run_all = False # If false, only run those experiments that have not yet run
data_setup.save_dataset = False
train.tuning_params = False
data_setup.BATCH_SIZE = 32
data_setup.TRAIN_SIZE = 0.61
data_setup.shuffle = False

if data_setup.TRAIN_SIZE <= 0.1:
    data_setup.save_dataset = True
data_setup.NUM_WORKERS = 1
data_setup.NUM_EPOCHS = 100
data_setup.MARGIN = 1.
data_setup.LEARNING_RATE = 1e-3
data_setup.PATIENCE = 10
data_setup.MIN_DELTA = 1e-6
#data_setup.RANDOM_SEED = None

data_setup.L1 = 32#128
data_setup.L2 = 64#256
data_setup.L3 = 64#512
data_setup.L4 = 32#256
data_setup.L5 = 64#128
data_setup.FC_UNITS1 = 512#1024
data_setup.FC_UNITS2 = 128
data_setup.FC_UNITS3 = 1024
data_setup.FC_UNITS4 = 512
data_setup.FC_UNITS5 = None

data_setup.device_nr = "2"
train.plot_matches = True
train.matching_test = False 
#data_setup.data_folder = "/scratch/antonmol/atm_new_dataset_sink49"
train.data_type = "krisha"

train_sinks = ["24", "80"]

data_folders = []
for sink in train_sinks:
    data_folders.append(f"/lustre/astro/krisha10/code/metadata_folder/sink{sink}")

data_setup.data_folder = data_folders
print("Training on:\n", data_setup.data_folder)
#data_setup.data_folder = "/lustre/astro/antonmol/atm_new_dataset_sink49"

data_setup.output_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs/_experiments"
#data_setup.dataloader_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders"


data_setup.distance_function = "cosine"  # cosine, euclid
data_setup.norm_type = "z_score" # minmax, z_score
DISTANCE_FUNCTION = data_setup.distance_function  # cosine, euclid

if train.data_type == "pkl":
    data_setup.md_names = ["ar_p", "ar_s", "mih", "m", "ds", "sep", "t"]
    # Set specific metadata configs
    all_md_configs = [
                        # ['ar_s', 'mih', 'm', 'ds', 'sep', 't'],
                        # ['m', 't'],
                        # ['mih', 'ds'],
                        # ['ar_s', 'mih', 'sep', 't'],
                        # ['ar_p', 'ar_s', 'mih', 'ds', 'sep', 't']
                    ]
    data_setup.normalizer_file = "metadata_stats_sink49.csv"
    train.target_sink = "165"
    train.target_projection = "3"
    target_normalizer_file = f"metadata_stats_sink{train.target_sink}.csv"

elif train.data_type == "krisha":
    data_setup.md_names = ["m_env", "ar", "temp", "m", "ds", "lum", "t"]
    # Set specific metadata configs
    all_md_configs = [
                        ["m_env", "ar", "temp", "m", "ds", "lum", "t"],
                        #["ar", "temp", "m", "lum", "t"],
                        #["m_env", "ds", "t"],
                        
                    ]
    data_setup.normalizer_file = "metadata_stats_sink24_krisha.csv"
    train.target_sink = "178"
    train.target_projection = "3"
    target_normalizer_file = f"metadata_stats_sink{train.target_sink}_krisha.csv"

train.target_normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/{target_normalizer_file}", index_col=0)


data_setup.normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/{data_setup.normalizer_file}", index_col=0)

data_setup.input_size = 512 # Change input image size if using cropping
data_setup.transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.CenterCrop(data_setup.input_size), # Crop image to 400x400 pixels 
    # transforms.RandomErasing(),
    # Add transform here
])

train.threshold = None
train.nsamp = 100
train.num_samples = 100

##### Experiment configurations #####
margins = [1.0] #[0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 5.0]

# Set kernel sizes for experimenting. Strides and paddings are automatically calculated
Ks = [3] #[3, 5, 7, 10, 20, 50]

top_combs = [] # Number of experiments for kps configurations
for i in range(len(Ks)):
    for j in range(len(Ks)):
        try:
            possible_combinations, combs = utils.get_kps(input_size=data_setup.input_size, 
                                                         n_configs=20, target_size=16,
            # NOTE: target_size is the size of the final conv2d layer and must be changed 
            #       accordingly, if encoder structure is altered! Otherwise model may
            #       get NaN gradients and training will be corrupted.
                                        ks=Ks, K2=Ks[i], K3=Ks[j])
            print(f"Possible combinations: {len(combs)}")
            top_combs.append(combs[0])
        except: 
            print(f"There are no possible combinations for K3={Ks[i]} and K4={Ks[j]}")

print("[[kernel_sizes], [paddings], [strides]]")
print(top_combs)

# set models for experimenting 
model_names = [
		        "SiameseNetwork_he_init",
                # "SiameseNetwork",  
                # "SiameseNetwork_ELU",
                #"SiameseNetwork_LeakyReLU", 
                "SiameseNetwork_fcs",
		        # "SiameseNetwork_he_init_batchnorm",
                "SiameseNetwork_batchnorm",
                # "SiameseNetwork_he_init_batchnorm_ELU",
                #"SiameseNetwork_extended",
                
               ] 



#all_md_configs = None

# ...or run for all configs / remaining configs
if len(all_md_configs) == 0:
    all_title_configs = [] # total number of combinations for 7 items is 2^7 - 1 = 127
    for r in range(1, len(data_setup.md_names) + 1):
        all_title_configs.extend(itertools.combinations(data_setup.md_names, r))
    all_md_configs = [list(config) for config in all_title_configs]

    # Get all mdconfigs / remaining configs 
    main_folder = os.listdir("/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs/_experiments/metadata")
    all_md_configs = utils.get_md_experiment(all_md_configs, main_folder, search="remaining")


print(f"Total number of metadata configurations: {len(all_md_configs)}")

data_sizes = [
                # 0.005,
                # 0.01,
                # 0.02,
                # 0.03,
                # 0.04,
                #  0.05,
                # 0.06,
                # 0.07,
                # 0.08,
                # 0.09,
                #  0.1, 
                # 0.12,
                # 0.14,
                # 0.16, 
                # 0.18, 
                # 0.2,
                # 0.22,
                # 0.24,
                #  0.5
                0.6
                ]

total_experiment_nr = len(top_combs)*len(model_names)*len(all_md_configs)*len(data_sizes)*len(margins)
print(f"\n[INFO] Running experiments for {len(top_combs)} configs over {len(model_names)} models and {len(all_md_configs)} metadata configurations\n and {len(data_sizes)} data_sizes and {len(margins)} margins.")
print(f"[INFO] - a total of {total_experiment_nr} experiments")


for m, margin in enumerate(margins):
    data_setup.MARGIN = margin
    # Iterate over kernel configurations
    for i in range(len(top_combs)):
        k1, pk1, k2, pk2, k3, pk3, k4, pk4, k5 = top_combs[i][0]
        p1, pp1, p2, pp2, p3, pp3, p4, pp4, p5 = top_combs[i][1]
        s1, ps1, s2, ps2, s3, ps3, s4, ps4, s5 = top_combs[i][2]

        models = [] # Set models
        if "SiameseNetwork" in model_names:
            models.append(model_builder.SiameseNetwork(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_ELU" in model_names:
            models.append(model_builder.SiameseNetwork_ELU(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_LeakyReLU" in model_names:
            models.append(model_builder.SiameseNetwork_LeakyReLU(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_he_init" in model_names:
            models.append(model_builder.SiameseNetwork_he_init(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_he_init_batchnorm" in model_names:
            models.append(model_builder.SiameseNetwork_he_init_batchnorm(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_batchnorm" in model_names:
            models.append(model_builder.SiameseNetwork_batchnorm(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_he_init_batchnorm_ELU" in model_names:    
            models.append(model_builder.SiameseNetwork_he_init_batchnorm_ELU(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, 
                                                fc_units2=data_setup.FC_UNITS2,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))
        if "SiameseNetwork_fcs" in model_names:
            models.append(model_builder.SiameseNetwork_fcs(l1=data_setup.L1, l2=data_setup.L2, l3=data_setup.L3, l4=data_setup.L4, l5=data_setup.L5, 
                                                fc_units1=data_setup.FC_UNITS1, fc_units2=data_setup.FC_UNITS2,
                                                fc_units3=data_setup.FC_UNITS3, fc_units4=data_setup.FC_UNITS4,
                                                fc_units5=data_setup.FC_UNITS5,
                                                k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
                                                s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                                p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,
                                                pk1=pk1, pk2=pk2, pk3=pk3, pk4=pk4,
                                                ps1=ps1, ps2=ps2, ps3=ps3, ps4=ps4,
                                                pp1=pp1, pp2=pp2, pp3=pp3, pp4=pp4
                                                ))

        # Iterate over models
        for j in range(len(models)):
            
            
            train.model = models[j]
            train.structure_dict = {
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
            
            
            

            
            
            # Iterate over metadata configurations
            for l, md_name in enumerate(all_md_configs):
                data_setup.md_names = md_name
                

                

                # Save configs to model dictionary
                for n, k in enumerate(top_combs[i][0]):
                    train.structure_dict[f"Ksize{n+1}"] = k
                for n, p in enumerate(top_combs[i][1]):
                    train.structure_dict[f"Padding{n+1}"] = p
                for n, s in enumerate(top_combs[i][2]):
                    train.structure_dict[f"Stride{n+1}"] = s
                

                
                # Run experiments
                # Total number of experiments:
                    # number of kernel configurations 
                    # * number of models
                    # * number of metadata configurations
                
                for d, data_size in enumerate(data_sizes):
                    data_setup.TRAIN_SIZE = data_size
                    if len(train_sinks) == 2:
                        data_setup.model_name = f"{model_names[j]}_batchsize{data_setup.BATCH_SIZE}_config{Ks[i]}_margin{data_setup.MARGIN}_md_{data_setup.md_names}_dsize{data_size}_sink{train_sinks[0]}_{train_sinks[1]}_not_random"
                    elif len(train_sinks) == 1:
                        data_setup.model_name = f"{model_names[j]}_batchsize{data_setup.BATCH_SIZE}_config{Ks[i]}_margin{data_setup.MARGIN}_md_{data_setup.md_names}_dsize{data_size}_sink{train_sinks[0]}_not_random"
          
                    train.structure_dict["model_name"] = data_setup.model_name
                    
                    current_exp_nr = (i+1)+(j+1)+(l+1)+(d+1)+(m+1)-4

                    print(f"\nRunning experiment {current_exp_nr} of {total_experiment_nr}")
                    print(f"\n[INFO] Starting training model {data_setup.model_name}\n")
                    print(f"Config: {top_combs[i]}\n")
                    print(f"Metadata: {md_name}")
                    print(f"Data size: {data_size}")
                    print(f"Margin: {margin}")

                    

                    if "SiameseNetwork_fcs" in data_setup.model_name:
                        train.structure_dict["fc3"] = data_setup.FC_UNITS3, 
                        train.structure_dict["fc4"] = data_setup.FC_UNITS4,
                        train.structure_dict["fc5"] = data_setup.FC_UNITS5,
                    

                    
                    try:
                        
                        train.main(train.model, data_setup.model_name)
                        
                    except Exception as e:
                        print(f"There was an error in training {data_setup.model_name}: {e}")
                        traceback.print_exc()
                

