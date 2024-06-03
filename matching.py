import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms, models
import data_setup, model_builder, engine, utils
import matplotlib.pyplot as plt
import argparse
import random
import pickle
import h5py
import sys
import gc


def get_target(target_dir, md_names, target_normalizer, projection="0", image_transform=None, batch_size=4, data_type="pkl"):

    if data_type == "pkl":
        target_file = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f"projection_{projection}" in f]
        target_md_file = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f"characteristics" in f]
        with open(target_file[0], 'rb') as f:
            target_image = pickle.load(f)
            target_image = np.array(target_image)
            target_image = (target_image - np.min(target_image)) / (np.max(target_image) - np.min(target_image))
            target_image = image_transform(target_image)
            target_image = target_image.unsqueeze(0)
            target_image = target_image.repeat(batch_size, 1, 1, 1) # Make sure the shape follows batch size of other data, typically 4

        with open(target_md_file[0], 'rb') as f:
            target_md_file = pickle.load(f)
            target_md, target_md_raw = data_setup.get_metadata(target_md_file, md_names, target_normalizer, data_type=data_type)
            target_md = torch.tensor(np.array(target_md)[0]).repeat(batch_size, 1)
            target_md_raw = torch.tensor(np.array(target_md_raw)[0]).repeat(batch_size, 1)
    
    elif data_type == "krisha":
        target_file = target_dir

        with h5py.File(target_file, "r") as hf:
            target_image = hf[f'intensity_projection_{projection}'][:]
            target_image = np.array(target_image)
            target_image = (target_image - np.min(target_image)) / (np.max(target_image) - np.min(target_image))
            target_image = image_transform(target_image)
            target_image = target_image.unsqueeze(0)
            target_image = target_image.repeat(batch_size, 1, 1, 1)

            target_md, target_md_raw = data_setup.get_metadata(target_file, md_names, target_normalizer, data_type=data_type)
            target_md = torch.tensor(np.array(target_md)[0]).repeat(batch_size, 1)
            target_md_raw = torch.tensor(np.array(target_md_raw)[0]).repeat(batch_size, 1)
    return target_image, target_md, target_md_raw

def get_model(output_dir,
              model_name,
              model=None,
              device=None
              ):
    ### Load model
    print("Loading model...")
    if not device:
        os.environ["CUDA_VISIBLE_DEVICES"]=data_setup.device_nr  # Choose which device to use (astro01 has 4 gpu's)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available

    checkpoint = torch.load(f"{output_dir}/{model_name}/{model_name}.pth")

    if not model:
        model_builder.input_size = 512
        model = model_builder.SiameseNetwork(l1=32, l2=32, l3=128, l4=64, l5=128, 
                                        fc_units1=512, 
                                        fc_units2=128)

    model_state_dict = model.state_dict()
    model.eval()
    model.load_state_dict(checkpoint)
    print("Model loaded")
    return model

def plot_results(output_dir, 
                 model_name,
                 md_names,
                 h5_match,
                 nsamp,
                 threshold,
                 target_sink,
                 target_snapshot,
                 rank,
                 ):
    
    
    truths, predictions, target_metadata, md2s, target_metadata_raw,  md2s_raw, \
        = utils.get_batch_data_hdf5(h5_match, ["truths", "predictions", "metadata1", "metadata2", "metadata1_raw", "metadata2_raw"])
    
    target_imgs, img2s = utils.get_batch_data_hdf5(h5_match, ["img1", "img2"])
    
    if rank == "best":
        if threshold is not None:
            ranked_pred_index = np.where(predictions >= threshold)[0]
            print(f"Matching {len(ranked_pred_index)} images above threshold {threshold}")
        else:
            ranked_pred_index = np.argsort(predictions)[::-1][:nsamp] # find the index of the best n predictions
            print(f"Matching {len(ranked_pred_index)} highest predictions")
        
    elif rank == "random":
        try:
            ranked_pred_index = np.random.choice(np.arange(len(predictions)), size=nsamp, replace=False)
        except:
            try:
                ranked_pred_index = np.random.choice(np.arange(len(predictions)), size=len(predictions), replace=False)
            except:
                print("Not enough datapoints for histogram of random metadata. Continuing without saving plot.")
                return

    ranked_predictions = predictions[ranked_pred_index]
    ranked_truths = truths[ranked_pred_index]
    
    avg_pred = np.mean(ranked_predictions)
    ranked_matches = img2s[ranked_pred_index]
    
    print("Rank:", rank)
    print("ranked predictions", ranked_predictions)

    bins = 30
    
    all_titles_old = ["accretion rate primary\n[Ms/yr]", "accretion rate secondary\n[Ms/s]", "mass in 100\n[g]",
                "mass\n[1/3000Ms]", "disk size\n[AU]", "separation\n[AU]", "t after formation\n[kyr]"]
    
    all_titles = ["accretion rate primary\n[$M_{\odot}$/yr]", "accretion rate secondary\n[$M_{\odot}$/yr]", "mass in 100\n[$M_{\odot}$]",
               "mass\n[$M_{\odot}$]", "disk size\n[AU]", "separation\n[AU]", "t after formation\n[kyr]"]

    title_mapping = {
        "ar_p": all_titles[0],
        "ar_s": all_titles[1],
        "mih": all_titles[2],
        "m": all_titles[3],
        "ds": all_titles[4],
        "sep": all_titles[5],
        "t": all_titles[6],
    }

    log_titles = []

    save_dir = os.path.join(output_dir, model_name, "matching")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_target = os.path.join(save_dir, f"metadata_distributions")
    if not os.path.exists(save_dir_target):
        os.makedirs(save_dir_target)
        
    save_file = os.path.join(save_dir_target, f"target_sink{target_sink}_{target_snapshot}_{rank}.png")
    plt.figure(figsize=(10,10))
    for i in range(len(md2s_raw[0][0])):
        metadata = md2s_raw[ranked_pred_index][:,0,i]
        target_md = target_metadata_raw[0][i]
        if md_names[i] == "ar_s": 
            metadata = np.array(metadata)/31557600 # convert units...
            target_md = np.array(target_md)/31557600
        if md_names[i] == "mih": 
            metadata = np.array(metadata)/1.989e33
            target_md = np.array(target_md)/1.989e33
        if md_names[i] == "m": 
            metadata = np.array(metadata)*3000
            target_md = np.array(target_md)*3000

        bins = 30
        if md_names[i] in log_titles:
            metadata = metadata[np.where(metadata != 0)]
            hist, new_bins = np.histogram(metadata, bins=bins)
            bins = np.logspace(np.log10(new_bins[0]), np.log10(new_bins[-1]), len(new_bins))
            
        plt.subplot(3,3,i+1); plt.hist(metadata, bins=bins, label="Best matches");
        plt.hist(metadata[0:30], bins=bins, color="r", label="Best 30 matches")
        
        high_count = np.histogram(metadata, bins=bins)[0].max()
        plt.vlines(target_md, 0, high_count, color="r", ls="--", label="Target")    
        if md_names[i] in log_titles:
            plt.xscale("log")        
        plt.title(title_mapping.get(md_names[i], "Not defined"))
        plt.legend()
    
    if rank == "best":
        plt.suptitle(f"Target: Sink {target_sink} snapshot {target_snapshot}\nAvg. pred: {avg_pred:.4f}\n from {len(ranked_predictions)} matches above threshold {threshold}")
    elif rank == "random":
        plt.suptitle(f"Target: Sink {target_sink} snapshot {target_snapshot}\nAvg. pred: {avg_pred:.4f}\n from {len(ranked_predictions)} random matches")
    
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()
    print(f"Saved image at {save_file}")


    # Plot metadata contributions
    print("Saving metadata contribution plots...")
    save_dir_comp = os.path.join(save_dir, "contributions")
    if not os.path.exists(save_dir_comp):
        os.makedirs(save_dir_comp)
    save_file_comp = os.path.join(save_dir_comp, f"target_sink{target_sink}_{target_snapshot}_{rank}.png")
    
    plt.figure(figsize=(10,10))
    for i in range(len(md2s[0][0])):
        metadata = md2s[ranked_pred_index][:,0,i]
        target_md = target_metadata[0][i]
        param = target_md*metadata
        plt.subplot(3,3,i+1)
        plt.hist(param, bins=50)
        plt.title(title_mapping.get(md_names[i], "Not defined"))
    plt.suptitle(f"Best {len(param)} matches above threshold:{threshold}")
    plt.tight_layout()


    # Plot comparisons
    print("Saving best matching images...")
    save_dir_comp = os.path.join(save_dir, "comparisons")
    if not os.path.exists(save_dir_comp):
        os.makedirs(save_dir_comp)
    save_file_comp = os.path.join(save_dir_comp, f"target_sink{target_sink}_{target_snapshot}_{rank}.png")
    
    plt.figure(figsize=(10,10))
    plt.subplot(3,3,1); plt.imshow(target_imgs[0][0])
    plt.title(f"Target image")
    for i, img in enumerate(ranked_matches):
        plt.subplot(3,3,i+2); plt.imshow(img[0])
        plt.title(f"Prediction: {ranked_predictions[i]:.3f}\nTruth: {ranked_truths[i]:.3f}")
        if i >= 7:
            break
    if rank == "best":
        plt.suptitle(f"Target: Sink {target_sink} snapshot {target_snapshot}\ncompared to best matches from sink 49")
    elif rank == "random":
        plt.suptitle(f"Target: Sink {target_sink} snapshot {target_snapshot}\ncompared to best of random matches from sink 49")
    plt.tight_layout()
    plt.savefig(save_file_comp)
    plt.close()
    print(f"Saved images at {save_file_comp}")






def run(output_dir,
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
        batch_size,
        device,
        nsamp,
        threshold,
        data_type="pkl"
        ):
    
    for target_snapshot in target_snapshots:
        if data_type == "pkl":
            target_dir = f"/lustre/astro/antonmol/atm_new_dataset_sink{target_sink}/dataset_{target_snapshot}"
        elif data_type == "krisha":
            target_dir = f"/lustre/astro/krisha10/code/metadata_folder/sink{target_sink}/output_{target_snapshot}_isink_{target_sink}_data.hdf5"
        
        print("Finding similarities for")
        print(f"target image: {target_dir}")
        print(f"and data from {test_dataloader_path}")

        target_image, target_md, target_md_raw = get_target(target_dir, 
                                                            md_names, 
                                                            target_normalizer, 
                                                            target_projection, 
                                                            transform, 
                                                            batch_size, 
                                                            data_type=data_type)
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.tensor(target_image)
        target_image = target_image.to(device).float()

        if not isinstance(target_md, torch.Tensor):
            target_md = torch.tensor(target_md)
        target_md = target_md.to(device).float()

        model = get_model(output_dir, model_name, model, device)

        ### Compute similarities
        print("Computing similarities...")

        h5_match = os.path.join(output_dir, model_name, "matching", "datafiles",
                                f"target_sink{target_sink}_{target_snapshot}.h5")
        os.makedirs(os.path.dirname(h5_match), exist_ok=True)

        with torch.no_grad(): # No need to store and calculate gradients! Significantly reduces memory usage
            for batch_id, batch in enumerate(test_dataloader):
                if batch_id > len(test_dataloader)-2:
                    break

                img2_batch = batch[1].to(device).float()
                
                MD2_tensor = batch[4][:,0,:].to(device).float()
                truth = F.cosine_similarity(target_md, MD2_tensor)
                out1, out2, prediction = model(target_image, img2_batch)
                with h5py.File(h5_match, "w") as hf:
                    test_params = {"img1": target_image, "img2": img2_batch,
                                "out1": out1, "out2": out2,
                                "truths": truth, 
                                "predictions": prediction,
                                "metadata1": target_md, "metadata2": batch[4].detach().cpu().numpy(),
                                "metadata1_raw": target_md_raw, "metadata2_raw": batch[6].detach().cpu().numpy(),
                                }
                    for key, value in test_params.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 0:
                            print(f"The tensor {key} is empty.")
                        elif value is None or (isinstance(value, str) and not value) or (isinstance(value, list) and not value):
                            print(f"The variable {key} is empty.")

                    utils.save_batch_data_hdf5(hf, test_params, batch_id)
                    
                del test_params, img2_batch

                torch.cuda.empty_cache()
                gc.collect()
                

        print(f"Saved outputs at {h5_match}")

        for rank in ["best", "random"]:
            plot_results(output_dir, 
                        model_name, 
                        md_names,
                        h5_match, 
                        nsamp,
                        threshold,
                        target_sink, 
                        target_snapshot, 
                        rank)


if __name__ == "__main__":

    TRAIN_SIZE = 0.62
    nsamp = 50 # number of top predictions to plot
    threshold = None # if threshold, overwrite nsamp and pick only matches with prediction above threshold
    BATCH_SIZE = 4

    target_sink = "91"
    target_snapshots = ["000444", "000555", "000655", "000788"]
    target_projection = "3"

    NUM_WORKERS = 1
    RANDOM_SEED = None
    dataloader_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/dataloaders"
    output_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs"
    data_folder = "/lustre/astro/antonmol/atm_new_dataset_sink49"
    model_name = "correct_data_test_run2"
    data_type = "pkl"
    DISTANCE_FUNCTION = data_setup.distance_function  # cosine, euclid
    md_names = [
                "ar_p", 
                "ar_s",
                "mih", 
                "m",
                "ds",
                "sep",
                "t",
                ]
    
    normalizer = pd.read_csv("/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_sink49.csv", index_col=0)
    target_normalizer = pd.read_csv(f"/lustre/astro/antonmol/learning_stuff/siamese_networks/metadata_stats_sink{target_sink}.csv", index_col=0)

    model = model_builder.SiameseNetwork_he_init_batchnorm(l1=32, l2=32, l3=128, l4=64, l5=128, 
                                        fc_units1=512, 
                                        fc_units2=128)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.CenterCrop(400), # Crop image to 400x400 pixels 
    
    ])
    # Setup target device
    os.environ["CUDA_VISIBLE_DEVICES"] = data_setup.device_nr  # Choose which device to use (astro01 has 4 gpu's)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available



    
    test_dataloader_path = os.path.join(dataloader_dir, \
            f"tsize{TRAIN_SIZE}_bsize{BATCH_SIZE}_test_{data_setup.input_size}_{data_setup.distance_function}.pth")
    
    test_dataloader = data_setup.create_dataloaders(
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
                                                    )[2]

    run(
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
        threshold,
        )



