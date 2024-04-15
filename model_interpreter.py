import matplotlib.pyplot as plt
import numpy as np
import os
import torch
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF

import model_builder, utils, engine

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import argparse

# import shap


attrib_anim = False

parser = argparse.ArgumentParser(description="Train a Siamese Network")

parser.add_argument("--l1", type=int, default=32, help="Number of channels at layers with l1 parameter. Check model for clarification.")
parser.add_argument("--l2", type=int, default=32, help="Number of channels at layers with l2 parameter. Check model for clarification.")
parser.add_argument("--l3", type=int, default=128, help="Number of channels at layers with l3 parameter. Check model for clarification.")
parser.add_argument("--l4", type=int, default=64, help="Number of channels at layers with l4 parameter. Check model for clarification.")
parser.add_argument("--l5", type=int, default=128, help="Number of channels at layers with l5 parameter. Check model for clarification.")
parser.add_argument("--fc1", type=int, default=512, help="Number of nodes in first FC layer. Check model for clarification.")
parser.add_argument("--fc2", type=int, default=128, help="Number of nodes in second FC layer, final output. Check model for clarification.")

parser.add_argument("--targets", type=utils.parse_targets, default=[0], help="The output node from the final FC layer to investigate in model interpretation")


parser.add_argument("--model_name", type=str, default="just_testing", help="Model name")

args = parser.parse_args()

L1 = args.l1
L2 = args.l2
L3 = args.l3
L4 = args.l4
L5 = args.l5
FC_UNITS1 = args.fc1
FC_UNITS2 = args.fc2

targets = args.targets

model_name = args.model_name


if __name__ == "__main__":

    model_builder.input_size = 400

    os.environ["CUDA_VISIBLE_DEVICES"]="2"  # Choose which device to use (astro01 has 4 gpu's)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use cuda (gpu) if available

    print(f"\nUsing {device} as device")
    print("Current device:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Device name", torch.cuda.get_device_name(torch.cuda.current_device()))

    output_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs"

    print("\nLoading model and data...")
    print("Model:", model_name)
    checkpoint = torch.load(f"{output_dir}/{model_name}/{model_name}.pth")

    model = model_builder.SiameseNetwork_he_init_batchnorm(l1=L1, l2=L2, l3=L3, l4=L4, l5=L5, 
                                        fc_units1=FC_UNITS1, 
                                        fc_units2=FC_UNITS2)

    model_state_dict = model.state_dict()
    model.eval()
    model.load_state_dict(checkpoint)
    

    h5_filename = os.path.join(output_dir, model_name, "batch_data.h5")
    truths, predictions, output1s, output2s, img1s, img2s, md1s, md2s = utils.get_batch_data_hdf5(h5_filename)
    print(targets)
    print(targets[0])
    if targets[0] > -1 and targets[0] < 998:
        print("Running FC2 method")
        layer = None
        method = "FC2"
        if attrib_anim == True:
            for n in [0, 1]:
                utils.animate_attr(output_dir, model_name, n, save=True)
    elif targets == [999]:
        print("Running layer method")
        targets = np.arange(0, L5-1)
        method = "layer"
        layer = model.encoder[0]

    # Loading saved output
    print("Loading saved output")
    utils.plot_attr(model, 
                    img1s, 
                    img2s, 
                    output_dir,
                    model_name,
                    device,
                    targets,
                    layer,
                    method)
