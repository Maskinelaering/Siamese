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


def get_target(target_dir, projection=0, transform=None, batch_size=4):

    target_file = [os.path.join(os.path.dirname(target_dir), f) for f in os.listdir(os.path.dirname(target_dir)) if f"projection_{projection}" in f]
    target_md_file = [os.path.join(os.path.dirname(target_dir), f) for f in os.listdir(os.path.dirname(target_dir)) if f"characteristics" in f]
    with open(target_file[0], 'rb') as f:
        target_image = pickle.load(f)
        target_image = transform(np.array(target_image))
        target_image = target_image.unsqueeze(0)
        target_image = target_image.repeat(batch_size, 1, 1, 1) # Make sure the shape follows batch size of other data, typically 4

    with open(target_md_file[0], 'rb') as f:
        target_md_file = pickle.load(f)
        target_md, target_md_raw = data_setup.get_metadata(target_md_file)
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
        os.environ["CUDA_VISIBLE_DEVICES"]="2"  # Choose which device to use (astro01 has 4 gpu's)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available

    checkpoint = torch.load(f"{output_dir}/{model_name}/{model_name}.pth")

    if not model:
        model_builder.input_size = 400
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
                 h5_match,
                 threshold,
                 ):
    truths, predictions, output1s, output2s, \
        img1s, img2s, md1s, md2s = utils.get_batch_data_hdf5(h5_match)

    #best_pred_index = np.argsort(predictions)[::-1]
    best_pred_index = np.where(predictions >= threshold)
    print("best predictions", predictions[best_pred_index[0]])
    # ### Plot results

    save_dir = os.path.join(output_dir, model_name, "matching")
    save_file = os.path.join(save_dir, "match.png")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    bins = int(0.3*len(best_pred_index[0]))
    
    if bins == 0: 
        bins = 1
    print("bins:", bins)
    print("threshold:", threshold)

    best_predictions = predictions[best_pred_index]
    avg_pred = np.mean(best_predictions)
    all_titles = ["accretion rate primary\n[Ms/yr]", "accretion rate secondary\n[Ms/s]", "mass in 100\n[g]",
                "mass\n[1/3000Ms]", "disk size\n[AU]", "separation\n[AU]", "t after formation\n[AU]"]
    
    plt.figure(figsize=(10,10))
    for i in range(len(md2s[0][0])):
        
        plt.subplot(3,3,i+1); plt.hist(md2s[best_pred_index][:,:,i], bins=bins);
        plt.vlines(md1s[0][i], 0, bins, color="r", ls="--")
        plt.title(all_titles[i])

    plt.suptitle(f"Avg. pred: {avg_pred}\n from {len(best_predictions)} predictions")
    plt.tight_layout()
    plt.savefig(save_file)
    print(f"Saved image at {save_file}")


def run(output_dir,
        model_name,
        model,
        test_dataloader,
        test_dataloader_path,
        target_dir,
        target_projection,
        transform,
        batch_size,
        device,
        threshold,
        ):

    print("Finding similarities for")
    print(f"target image: {target_dir}")
    print(f"and data from {test_dataloader_path}")
    target_image, target_md, target_md_raw = get_target(target_dir, target_projection, transform, batch_size)
    
    model = get_model(output_dir, model_name, model, device)

    ### Compute similarities
    print("Computing similarities...")

    h5_match = os.path.join(output_dir, model_name, "match_data.h5")
    os.makedirs(os.path.dirname(h5_match), exist_ok=True)

    with h5py.File(h5_match, "w") as hf:
        for batch_id, batch in enumerate(test_dataloader):
            if batch_id > len(test_dataloader)-2:
                break

            #img1_batch = batch[0]    
            img2_batch = batch[1]
            
            #md1_batch = batch[3]
            md2_batch = np.array(batch[4])
            md2_batch_raw = np.array(batch[6])

            MD1_tensor = target_md 
            MD2_tensor = torch.tensor(md2_batch)[:,0,:]

            truth = F.cosine_similarity(MD1_tensor, MD2_tensor)
            out1, out2, prediction = model(target_image.to(device).float(), img2_batch.to(device).float())
            
            test_params = {"img1": target_image, "img2": img2_batch,
                        "out1": out1, "out2": out2,
                        "truths": truth, 
                        "predictions": prediction,
                        "metadata1": target_md_raw, "metadata2": md2_batch_raw,
                        }
            utils.save_batch_data_hdf5(hf, test_params, batch_id)
    print(f"Saved outputs at {h5_match}")

    plot_results(output_dir, model_name, h5_match, threshold)



















