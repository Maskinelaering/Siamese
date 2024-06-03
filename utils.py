import torch
import torch.nn as nn
#import torchsummary
from torchsummary import summary
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
import tqdm as tqdm
import h5py
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import itertools

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
import glob
from sklearn import metrics, preprocessing

from sklearn.metrics.pairwise import cosine_similarity

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import Saliency, DeepLift, NoiseTunnel
from captum.attr import visualization as viz


##### ----------------- Step 4: Utilities --------------------

def parse_targets(targets):
    try:
        targets = [int(target.strip()) for target in targets.split(',')]
        return targets
    except ValueError:
        raise argparse.ArgumentTypeError("Targets must be integers")

def save_model(model: nn.Module,
               output_dir: str,
               model_name: str,
               structure_df):
    """
    Saves a PyTorch model to a target directory.

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
    output_dir_path = Path(model_dir_path)
    output_dir_path.mkdir(parents=True,
                         exist_ok=True)
    
    model_savename = "{}/{}.pth".format(output_dir_path, model_name)  # Name of saved model
    torch.save(obj=model.state_dict(),
             f=model_savename)

    if structure_df is not None:
        print(f"[INFO] Saved model to: {output_dir_path}")
        summary_filename = "{}/{}_summary.txt".format(output_dir_path, model_name)
        
        structure_df.to_csv(summary_filename, sep='\t', index=True, header=False)
        

def save_batch_data_hdf5(hf, params: dict, batch_id: int):
    
    """
    Used to save data during training and testing. 
    """

    batch_group = hf.create_group(f"{batch_id}")

    for name, param in params.items():
        if torch.is_tensor(param):
            batch_group.create_dataset(f"{name}", data=param.detach().cpu().numpy())
        else:
            batch_group.create_dataset(f"{name}", data=param)


def get_batch_data_hdf5(h5_filename, param_names):

    """
    Returns specified items stored in h5_filename. Items must be specified with true keys in a list 
    called param_names. Returns items in the same order as in param_names list.
    
    Example usage:
    truths, predictions = utils.get_batch_data_hdf5(h5_filename, param_names = ["truths", "predictions"])
    """

    data = {name: [] for name in param_names}

    with h5py.File(h5_filename, "r") as hf:
        for batch_id, batch_group in hf.items():
            for name in param_names:
                try: 
                    batch_group[name]
                except:
                    print(f"Parameter {name} not in file {h5_filename}. \n Parameters: {batch_group.keys()}")
                    break
                if name in ["truths", "predictions"]:
                    data[name].append(batch_group[name][:].ravel())
                else:
                    data[name].append(batch_group[name][:])

    # Convert lists to numpy arrays
    try:
        for name, array_list in data.items():
           
            # Check if the parameter is two-dimensional
            if len(array_list[0].shape) > 1:
                data[name] = np.vstack(array_list)
            else:
                data[name] = np.concatenate(array_list)
    except:
        print("\nSome of the saved hdf5 file items are empty. Skipping process. This might lead to further errors.\n")
        return 

    # Return only the specified parameters
    return tuple(data[name] for name in param_names)




def create_clustering_plot(h5_filename, output_dir, model_name, id=None):
    """
    Creates plot of truths versus predictions for 
    1. The direct comparisons: 
        The true and predicted values for similarity of each image pair is compared
    2. The ranked comparisons:
        The lowest true value is compared to the lowest predicted value, giving an overview
        of how the model ranks pairs based on similarity compared to the actual given similarity labels.

    """
    truths, predictions = get_batch_data_hdf5(h5_filename, ["truths", "predictions"])
    
    save_dir = os.path.join(output_dir, model_name, "images")
    save_name = os.path.join(output_dir, model_name, "images")
    output_dir_path = Path(save_name)
    if not os.path.exists(output_dir_path):
        output_dir_path.mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(save_dir, f"clustering.png")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Direct comparison
    axs[0, 0].plot(truths, predictions, '.', label="Test predictions vs. true values")
    axs[0, 0].plot(np.linspace(np.min(truths), np.max(truths), 100), 
                np.linspace(np.min(truths), np.max(truths), 100),
                ls="--", color="k", label="Perfect score")
    axs[0, 0].set_xlabel("Truths")
    axs[0, 0].set_ylabel("Predictions")
    axs[0, 0].legend()
    axs[0, 0].set_title("Direct comparison")

    axs[1, 0].plot(truths, predictions, '.', label="Test predictions vs. true values")
    axs[1, 0].plot(np.linspace(np.min(truths), np.max(truths), 100), 
                np.linspace(np.min(truths), np.max(truths), 100),
                ls="--", color="k", label="Perfect score")
    axs[1, 0].set_xlim(0.8, 1.0)
    axs[1, 0].set_ylim(0.8, 1.0)
    axs[1, 0].set_xlabel("Truths")
    axs[1, 0].set_ylabel("Predictions")

    # Ranked comparison
    truths_sorted = truths[np.argsort(truths)]
    predictions_sorted = predictions[np.argsort(predictions)]
    axs[0, 1].plot(truths_sorted, predictions_sorted, '.', label="Test predictions vs. true values")
    axs[0, 1].plot(np.linspace(np.min(truths), np.max(truths), 100), 
                np.linspace(np.min(truths), np.max(truths), 100),
                ls="--", color="k", label="Perfect score")
    axs[0, 1].set_xlabel("Truths")
    axs[0, 1].set_ylabel("Predictions")
    axs[0, 1].set_title("Ranked comparison")

    axs[1, 1].plot(truths_sorted, predictions_sorted, '.', label="Test predictions vs. true values")
    axs[1, 1].plot(np.linspace(np.min(truths), np.max(truths), 100), 
                np.linspace(np.min(truths), np.max(truths), 100),
                ls="--", color="k", label="Perfect score")
    axs[1, 1].set_xlim(0.8, 1.0)
    axs[1, 1].set_ylim(0.8, 1.0)
    axs[1, 1].set_xlabel("Truths")
    axs[1, 1].set_ylabel("Predictions")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    
def calculate_mse_from_hdf5(h5_filename):
    "Used to calculate the mean squared error of the truths vs. predictions"
    truths, predictions = get_batch_data_hdf5(h5_filename, ["truths", "predictions"])
    print("Truth/prediction datapoints:", len(truths))
    if len(truths) >= 30:
        print("First 30 truths:", truths[0:30])
        print("First 30 predictions", predictions[0:30])

    mse = metrics.mean_squared_error(truths, predictions)
    return mse


def calculate_accuracy_from_hdf5(h5_filename, error=0.1, low=0.9, high=1.0):
    "Used to calculate the accuracy of the truths vs. predictions"
    truths, predictions = get_batch_data_hdf5(h5_filename, ["truths", "predictions"])
    
    truths = truths[np.where((predictions >= low) & (predictions <= high))]
    predictions = predictions[np.where((predictions >= low) & (predictions <= high))]
    
    true_scores = []
    for t, p in zip(truths, predictions):
        if abs(t-p) < error:
            score = 1
        else:
            score = 0
        true_scores.append(score)
    acc_mean = np.sum(true_scores) / len(true_scores)  
    acc_sum = np.sum(true_scores)

    return acc_mean, acc_sum


def plot_training_evolution(h5_training_stats, output_dir, model_name):
    training_loss, validation_loss = [], []
    first_train_loss, second_train_loss = [], []
    first_val_loss, second_val_loss = [], []
    
    
    with h5py.File(h5_training_stats, "r") as hf:
        hf_group = hf["all_batches"]
        train_mse = hf_group["train_mse"][()]
        val_mse = hf_group["validation_mse"][()]
        epochs = hf_group["epoch"][()]

        training_loss = hf_group["training_loss"][()]
        first_train_loss = hf_group["first_train_loss"][()]
        second_train_loss = hf_group["second_train_loss"][()]
        validation_loss = hf_group["validation_loss"][()]
        first_val_loss = hf_group["first_val_loss"][()]
        second_val_loss = hf_group["second_val_loss"][()]

        try:
            train_reconstruction_loss = hf_group["train_reconstruction_loss"][()]
            validation_reconstruction_loss = hf_group["validation_reconstruction_losses"][()]
            SA = True
        except:
            SA = False

    plt.figure()
    plt.plot(epochs, training_loss, color="b", label="Training loss")
    plt.plot(epochs, first_train_loss, '--', color="b", alpha=0.7, label="First part train")
    plt.plot(epochs, second_train_loss, '-.', color="b", alpha=0.7, label="Second part train")
    plt.plot(epochs, validation_loss, color="r", label="Validation loss")
    plt.plot(epochs, first_val_loss, '--', color="r", alpha=0.7, label="First part val")
    plt.plot(epochs, second_val_loss, '-.', color="r",  alpha=0.7, label="Second part val")
    if SA:
        plt.plot(epochs, train_reconstruction_loss, '-x', color="b", alpha=0.7, label="Autoencoder train loss")
        plt.plot(epochs, validation_reconstruction_loss, '-x', color="b", alpha=0.7, label="Autoencoder val loss")
    plt.legend()
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model: {model_name}")
    save_dir = os.path.join(output_dir, model_name, "images")
    if not os.path.exists(save_dir):
          save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_dir}/training_evol.png")
    plt.close()
    print(f"[INFO] Saved training evolution plot at {save_dir}/training_evol.png")

    plt.figure()
    plt.plot(epochs, train_mse, color="b", label="Train MSE")
    plt.plot(epochs, val_mse, color="r", label="Validation MSE")
    plt.legend()
    plt.title(f"Model: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(f"{save_dir}/MSEs.png")
    plt.close()




def plot_attr(model, 
              img1s, 
              img2s, 
              output_dir,
              model_name,
              device,
              targets,
              layer=None,
              method="FC2",
              sign="all",
              summing=False,
              plot=True,
              ):

    """
    This function does model interpretation with the specified method of Integrated Gradients,
    see: https://captum.ai/api/integrated_gradients.html 

    Args:
        model: the model that is investigated
        img1s: a list of the first input images with the shape [batch_size, channels, height, width]
                Important: must match the shape of input images used to train the model
        img2s: a list of the second input images with same shape as img1s
        output_dir: where the outputs are stored
        model_name: name of model - used to import model and save outputs in model folder
        device: cuda or cpu
        targets: list of integers. Defines which nodes of the layer are analysed.
                Maximum value is the number of nodes-1 of the layer
        layer: type: model.<layer_name>. Used for method="layer".
                Example: model.encoder[0]
        method: specifies the method of analysis.
            "FC2": analyses the target nodes of the final fully-connected layer, i.e. the final output.
            "layer"; analyses the target nodes of the specified layer.
    """
    
    

    img1 = torch.tensor(img1s[0:1], requires_grad=True).to(device)
    img2 = torch.tensor(img2s[0:1], requires_grad=True).to(device)
    combined_images = torch.cat((img1, img2), dim=1)
    
    vis_folder = os.path.join(output_dir, model_name, "images", "attributions")
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)


    def running(image_number, target, sign):
        """
        This function runs the integrated gradients analysis for one of the input images
        and will run twice to analyse both images.
        """

        checkpoint_path = f"{output_dir}/{model_name}/{model_name}.pth" 
        if os.path.exists(checkpoint_path): # only load trained model if exists..
            checkpoint = torch.load(checkpoint_path)
            model_state_dict = model.state_dict()
            model.eval()
            model.load_state_dict(checkpoint)
        

        def combined_forward(imgs, model):
            output1, output2, prediction = model(imgs[:, :1, :, :], imgs[:, 1:, :, :])

            if image_number == 0:
                output = output1
            elif image_number ==1:
                output = output2
            
            return output

        
        if method == "FC2":
            grad = IntegratedGradients(combined_forward)
            save_name = os.path.join(vis_folder, f"attr_{method}_tar{target}_{sign}_img{image_number}.png")
            if summing == True:
                summed = True
            else:
                summed = False
        elif method == "layer":
            grad = LayerIntegratedGradients(combined_forward, layer,
                                            multiply_by_inputs=False
                                            )
            summed = True
            save_name = os.path.join(vis_folder, f"attr_{method}_{layer}_{sign}_summed_img{image_number}.png")
        
        if summed == True:
            # Compute attributions for each target
            attributions = []
            for target_val in targets:
                target_val = int(target_val)
                attr = grad.attribute(combined_images,
                                    additional_forward_args=(model,),
                                    target=target_val,  
                                    )
                attribution = attr.cpu().detach().numpy()[0]
                attributions.append(attribution)
            attrib = np.sum(attributions, axis=0)
            attrib = np.transpose(attrib, (1, 2, 0))
        else:
            attr = grad.attribute(combined_images,
                            additional_forward_args=(model,),
                            # baselines=baselines,  # Initial value of the integration (?)
                            target=target,  # Which of the output elements should be investigated.   
                            )
            attributions = attr.cpu().detach().numpy()[0]
            attrib = np.transpose(attributions, (1, 2, 0))


        if image_number == 0:
            original_image = img1.cpu().detach().numpy()[0]
        elif image_number == 1:
            original_image = img2.cpu().detach().numpy()[0]

        origin = np.transpose(original_image, (1, 2, 0))
        
        if plot == True:
            if np.isnan(attrib).any():
                print("WARNING: Attributions contains nan values")
                print("Using matplotlib instead (NOTE: image saving needs correction)")
                plt.figure()
                plt.imshow(origin, alpha=0.5)
                plt.imshow(attrib, alpha=0.5)
                plt.savefig(save_name)
                plt.close()

                plt.figure()
                plt.imshow(attrib)
                plt.savefig(save_name)
                plt.close()
            else:    
                import matplotlib as mpl
                c_white = mpl.colors.colorConverter.to_rgba('white',alpha = 0)
                c_black= mpl.colors.colorConverter.to_rgba('green',alpha = 1)
                cmap_rb = mpl.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_black],64)

                c_white = mpl.colors.colorConverter.to_rgba('red',alpha = 1)
                c_black= mpl.colors.colorConverter.to_rgba('green',alpha = 1)
                cmap_rg = mpl.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_black],64)
                if method == "layer":
                    figure, axis = viz.visualize_image_attr(attrib, origin, method="blended_heat_map", sign=sign,
                                                        alpha_overlay=0.3, show_colorbar=True,# cmap=cmap_rb,
                                                        title=f"Overlayed Integrated Gradients\nImage: {image_number}\nMethod: {method}\nTarget layer: \n{layer}",
                                        )
                elif method == "FC2":
                    figure, axis = viz.visualize_image_attr(attrib, origin, method="blended_heat_map", sign=sign,
                                                            alpha_overlay=0.3, show_colorbar=True,# cmap=cmap_rb,
                                                            title=f"Overlayed Integrated Gradients\nImage: {image_number}\nMethod: {method}\nTarget(s): {target}",
                                            )
                axis.set_xlim(150, 250)
                axis.set_ylim(150, 250)
                
                figure.savefig(save_name)
                print(f"Saved attributions figure at {save_name}")
                plt.close()
        return attrib

    # Running for each input image
    attribs0 = []
    attribs1 = []
    if method == "layer":
        attrib0 = running(0, targets, sign)
        attrib1 = running(1, targets, sign)
        attribs0.append(attrib0)
        attribs1.append(attrib1)
    else:
        if summing == True:
            attrib0 = running(0, targets, sign)
            attrib1 = running(1, targets, sign)
            attribs0.append(attrib0)
            attribs1.append(attrib1)
        else:
            for target in targets:
                target = int(target)
                attrib0 = running(0, target, sign)
                attrib1 = running(1, target, sign)
                attribs0.append(attrib0)
                attribs1.append(attrib1)
    return attribs0, attribs1



def animate_attr(output_dir, model_name, n, method, save=False):
    # Get a list of all image files in the directory containing "imgn" in their name
    image_dir = os.path.join(output_dir, model_name, "images", "attributions")
    if method == "layer":
        image_files = glob.glob(os.path.join(image_dir, f"*summed_img{n}*.png"))
    elif method == "FC2":
        image_files = glob.glob(os.path.join(image_dir, f"*img{n}*.png"))
    image_files.sort()
    print(image_files)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.axis('off')

    # Display the initial image
    image_index = 0
    image_path = os.path.join(image_dir, image_files[image_index])
    image = plt.imread(image_path)
    img = ax.imshow(image)

    # Define a function to update the displayed image
    def animate(index):
        image_path = os.path.join(image_dir, image_files[index])
        image = plt.imread(image_path)
        img.set_data(image)


    ani = FuncAnimation(fig, animate, frames=len(image_files), interval=500)
    html_output = HTML(ani.to_jshtml())

    clear_output(wait=True)
    

    if save == True:
        save_name = os.path.join(image_dir, f'attr_vis_anim_img{n}.gif')
        ani.save(save_name, writer='pillow', dpi=300)

    display(html_output)



def plot_attr_homemade(model, img1s, img2s, plot_imgs, output_dir, model_name, device,
                       targets, layer, method="FC2", sign="all", summing=True, order="trained"):
    print("\nComputing integrated gradients...")
    if method == "layer":
        print(f"... on layer {layer}")
    elif method == "FC2":
        print(f"... on FC2 layer node(s): {targets}")

    if isinstance(targets[0], np.ndarray):
            targets = targets[0]
    #print("TARGETS", targets)
    if len(targets) > 1:
        summing = True

    try:
        attribs0, attribs1 = plot_attr(model, 
                img1s, 
                img2s, 
                output_dir,
                model_name,
                device,
                targets,
                layer,
                method,
                sign,
                summing,
                plot=False
                )
    
        vis_folder = os.path.join(output_dir, model_name, "images", "attributions")
        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)
        c_white = mpl.colors.colorConverter.to_rgba('white',alpha = 0)
        c_black= mpl.colors.colorConverter.to_rgba('green',alpha = 1)
        c_red = mpl.colors.colorConverter.to_rgba('red',alpha = 0)
        c_purple = mpl.colors.colorConverter.to_rgba('purple',alpha = 1)
        cmap_rb = mpl.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white, c_black],64)
        cmap_ry = mpl.colors.LinearSegmentedColormap.from_list('ry_cmap', [c_red, c_purple], 64)

        for image in [0,1]:
            if image == 0:
                attr = attribs0[0][:, :, 0]#.sum(axis=2)
            elif image == 1:
                attr = attribs1[0][:, :, 1]#.sum(axis=2)

            plot_img = plot_imgs[image]

            plt.figure(figsize=(7,7))
            plt.subplot(2, 2, 1)
            #img = plt.imshow(attr, cmap="Oranges", norm=LogNorm(), alpha=0.9)#, norm=LogNorm(vmin=0.1, vmax=100))
            plt.imshow(plot_img, cmap="Greys", norm=LogNorm(), alpha=1)
            #plt.colorbar(img)
            # plt.xlim(200, 300)
            # plt.ylim(200, 300)

            plt.subplot(2, 2, 2)
            img = plt.imshow(attr, cmap=cmap_ry, norm=LogNorm(), alpha=1.)#, norm=LogNorm(vmin=0.1, vmax=100))
            plt.imshow(plot_img, cmap="Greys", norm=LogNorm(),  alpha=0.8)
            #plt.colorbar(img)
            # plt.xlim(200, 300)
            # plt.ylim(200, 300)

            plt.subplot(2, 2, 3)
            #img = plt.imshow(attr, cmap="Oranges", norm=LogNorm(), alpha=0.9)#, norm=LogNorm(vmin=0.1, vmax=100))
            plt.imshow(plot_img, cmap="Greys", norm=LogNorm(), alpha=1)
            #plt.colorbar(img)
            smid = int(len(plot_img[0])/2) 
            s20 = 0.1 * len(plot_img[0])
            plt.xlim(smid-s20, smid+s20)
            plt.ylim(smid+s20, smid-s20)

            plt.subplot(2, 2, 4)
            img = plt.imshow(attr, cmap=cmap_ry, norm=LogNorm(), alpha=1.)#, norm=LogNorm(vmin=0.1, vmax=100))
            plt.imshow(plot_img, cmap="Greys", norm=LogNorm(), alpha=0.8)
            #plt.colorbar(img)
            plt.xlim(smid-s20, smid+s20)
            plt.ylim(smid+s20, smid-s20)

            if method == "layer":
                plt.suptitle(f"Integraded gradients for \nlayer: {layer}")
                save_name = os.path.join(vis_folder, f"{order}_attr_{method}_{layer}_{sign}_img{image}.png")
            elif method == "FC2":
                
                if len(targets) > 1:
                    save_name = os.path.join(vis_folder, f"{order}_attr_{method}_nodes[{targets[0]}:{targets[-1]}]_{sign}_summed_img{image}.png")
                    plt.suptitle(f"Integraded gradients for output layer nodes:\n[{targets[0]}:{targets[-1]}]")
                else:
                    save_name = os.path.join(vis_folder, f"{order}_attr_{method}_node{targets[0]}_{sign}_summed_img{image}.png")
                    plt.suptitle(f"Integraded gradients for output layer node {targets[0]}")
            plt.tight_layout()
            
            
            plt.savefig(save_name)
            plt.close()
            print(f"Saved image at {save_name}")
            #plt.show()
    except Exception as e:
        print("There was an error in creating attributions for untrained model:")
        print(e)



def plot_x_best(h5_filename, output_dir, model_name, sorter="truths", order="low"):
    # Initialize lists to store the lowest/highest truth values and their corresponding indices
    values = []
    indices = []
    values_other = []

    if sorter == "truths":
        main_sorter = "truths"
        other = "predictions"
    elif sorter == "predictions":
        main_sorter = "predictions"
        other = "truths"
    
    # Iterate over all groups in the HDF5 file
    with h5py.File(h5_filename, "r") as hf:
        for group_key in hf.keys():
            group = hf[group_key]
            main_dataset = group[main_sorter]
            other_dataset = group[other]
            
            main_array = np.array(main_dataset)
            other_array = np.array(other_dataset)
            group_indices = np.argsort(main_array)

            group_values = main_array[group_indices]
            group_values_other = other_array[group_indices]
            
            # Update lists based on the order
            if order == "high":
                for value, other_value, index in zip(group_values, group_values_other, group_indices):
                    if len(values) < 10 or value > min(values):
                        if len(values) == 10:
                            min_index = np.argmin(values)
                            del values[min_index]
                            del indices[min_index]
                            del values_other[min_index]
                        
                        values.append(value)
                        values_other.append(other_value)
                        indices.append((group_key, index))
            elif order == "low":
                for value, other_value, index in zip(group_values, group_values_other, group_indices):
                    if len(values) < 10 or value < max(values):
                        if len(values) == 10:
                            max_index = np.argmax(values)
                            del values[max_index]
                            del indices[max_index]
                            del values_other[max_index]
                        
                        values.append(value)
                        values_other.append(other_value)
                        indices.append((group_key, index))

    # Retrieve the corresponding image pairs for the lowest/highest values
    h5_image_file = os.path.join(output_dir, model_name, "batch_images.h5")
    with h5py.File(h5_image_file, "r") as hf:
        plt.figure(figsize=(20, 4))
        for i, (group_key, index) in enumerate(indices):
            group = hf[group_key]
            img1_dataset = group["img1"]
            img2_dataset = group["img2"]
            img1 = img1_dataset[index][0]
            img2 = img2_dataset[index][0]
            plt.subplot(2, 10, i + 1)
            plt.imshow(img1)
            plt.title(f"{main_sorter}: {values[i]:.2f} \n{other}: {values_other[i]:.2f}\nImg1")
            plt.axis('off')
            
            plt.subplot(2, 10, 10 + i + 1)
            plt.imshow(img2)
            plt.title("Img2")
            plt.axis('off')

            if sorter == "truths":
                if order == "high":
                    plt.suptitle("10 most similar images according to true similarities")
                    save_name = os.path.join(output_dir, model_name, "images", "best10_truths.png")
                elif order == "low":
                    plt.suptitle("10 least similar images according to true similarities")
                    save_name = os.path.join(output_dir, model_name, "images", "worst10_truths.png")
            elif sorter == "predictions":
                if order == "high":
                    plt.suptitle("10 most similar images according to predicted similarities")
                    save_name = os.path.join(output_dir, model_name, "images", "best10_predictions.png")
                elif order == "low":
                    plt.suptitle("10 least similar images according to predicted similarities")
                    save_name = os.path.join(output_dir, model_name, "images", "worst10_predictions.png")
        plt.tight_layout()
        
        plt.savefig(save_name)
        plt.close()



def plot_gradient_evol(model_parameter_df,
                       output_dir,
                       model_name,): 
    mean_gradient_norms = model_parameter_df.mean()
    std_gradient_norms = model_parameter_df.std()

    threshold = 3 * std_gradient_norms  
    
    plt.figure(figsize=(10, 6))
    for column in model_parameter_df.columns:
        plt.plot(model_parameter_df.index, model_parameter_df[column], label=column)

    # Identify epochs with large increases or decreases
    large_increases = model_parameter_df[model_parameter_df.diff() > threshold]
    large_decreases = model_parameter_df[model_parameter_df.diff() < -threshold]

    # Add markers for large increases
    for index, row in large_increases.iterrows():
        plt.scatter([index] * len(row), row.values, color='red', marker='^')

    # Add markers for large decreases
    for index, row in large_decreases.iterrows():
        plt.scatter([index] * len(row), row.values, color='blue', marker='v')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    if not any((model_parameter_df < 0)):
        plt.yscale('log')
    plt.title('Gradient Norms over Epochs')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    save_file = os.path.join(output_dir, model_name, "images", "gradient_evolution")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()



def get_kps(input_size, n_configs, target_size, ks,
            K2=None, K3=None, P2=None, P3=None, S2=None, S3=None):
    def calculate_input_size(input_size, kernel_sizes, paddings, strides):
        for i, (k, p, s) in enumerate(zip(kernel_sizes, paddings, strides)):
            if (i+1)%2 == 0:
                input_size = ((input_size + 2*p - (k-1) - 1) // s) + 1
            else: 
                input_size = ((input_size + 2*p - k) // s) + 1
        return input_size

    init_kernel_sizes = [3, None, None, 3, 3]
    init_paddings = [1, None, None, 1, 1]
    init_strides = [1, None, None, 1, 1]
    possible_combinations = []
    for k2 in ks:
        for k3 in ks:
            for p2 in range(1, n_configs + 1):
                for p3 in range(1, n_configs + 1):
                    for s2 in range(1, 5):
                        for s3 in range(1, 5):
                            kernel_sizes = [3, 3, k2, 3, k3, 3, 3, 3, 3]
                            paddings = [1, 1, p2, 1, p3, 1, 1, 1, 1]
                            strides = [1, 2, s2, 2, s3, 2, 1, 2, 1]
                            if calculate_input_size(input_size, kernel_sizes, paddings, strides) == target_size:
                                possible_combination = [kernel_sizes, paddings, strides]
                                #print(possible_combination)
                                possible_combinations.append(possible_combination)


    specified_kernel_sizes = [3, 3, K2, 3, K3, 3, 3, 3, 3]
    specified_paddings = [1, 1, P2, 1, P3, 1, 1, 1, 1]
    specified_strides = [1, 2, S2, 2, S3, 2, 1, 2, 1]
    combs = []
    for combination in possible_combinations:
        # Check if the combination matches the specified partial combination
        if (all(x is None or x == y for x, y in zip(specified_kernel_sizes, combination[0])) and
            all(x is None or x == y for x, y in zip(specified_paddings, combination[1])) and
            all(x is None or x == y for x, y in zip(specified_strides, combination[2]))):
            combs.append(combination)


    return possible_combinations, combs




def plot_accuracy(h5_filename, output_dir, model_name, lower_limit=0.9, upper_limit=1.0, bins=10, plot=True):
    
    truths, predictions = get_batch_data_hdf5(h5_filename, ["truths", "predictions"])

    top_truths = truths[np.where((predictions >= 0.9) & (predictions <= 1.0))]
    top_predictions = predictions[np.where((predictions >= 0.9) & (predictions <= 1.0))]
    
    truths = truths[np.where((predictions >= lower_limit) & (predictions <= upper_limit))]
    predictions = predictions[np.where((predictions >= lower_limit) & (predictions <= upper_limit))]
    
    errors = np.linspace(0, 1.0, bins+1)

    if not np.any((np.array(np.round(truths * 100) / 100) != 1.) & (np.array(np.round(truths * 100) / 100) != -1.)):
        #print("Only 1s and -1s for truths")
        return np.zeros(len(errors)), np.zeros(len(errors))

 
    def scoring(errors, truths, predictions):
        acc_sums = []
        acc_means = []
        for i, error in enumerate(errors):
            true_scores = []
            for t, p in zip(truths, predictions):
                if i == 0 and abs(t-p) < error:
                    score = 1
                elif abs(t-p) < error and abs(t-p) >= errors[i-1]:
                    score = 1
                else:
                    score = 0
                true_scores.append(score)
            acc_mean = np.sum(true_scores) / len(true_scores)  
            acc_means.append(acc_mean)
            acc_sums.append(np.sum(true_scores))
        return acc_means, acc_sums

    acc_means, acc_sums = scoring(errors, truths, predictions)

    if plot == True:
        plt.figure(figsize=(8,5))

        bars1 = plt.bar(np.linspace(-0.1, 0.9, bins+1), acc_sums, width=0.1, align="edge", label="Number of correct guesses within error")
        for i, rect in enumerate(bars1):
            height = rect.get_height()
            if height != 0:
                total_sum = np.sum(acc_sums)
                percentage = height / total_sum * 100
                plt.text(rect.get_x() + rect.get_width() / 2, height, f'{percentage:.2f}%',
                        ha='center', va='bottom')
                
        if not lower_limit == 0.9 and upper_limit == 1.0:
            top_acc_means, top_acc_sums = scoring(errors, top_truths, top_predictions)
            top_total_sum = np.sum(top_acc_sums)
            bars2 = plt.bar(np.linspace(-0.1, 0.9, bins+1), top_acc_sums, width=0.1, align="edge", alpha=0.5, label="Top predictions above 0.9")
            for i, rect in enumerate(bars2):
                height = rect.get_height()
                if height != 0:
                    percentage = height / top_total_sum * 100
                    if percentage >= 1.0:
                        plt.text(rect.get_x() + rect.get_width() / 2, height, f'{percentage:.2f}%',
                                ha='center', va='bottom', color='white')
        plt.legend()
        plt.xlim(-0.05, 1.05)
        plt.xlabel("Error")
        plt.ylabel("Counts")
        plt.title(f"Model accuracy as function of error for\npredictions in limit [{lower_limit}:{upper_limit}]\n\nExperiment: {model_name}")
        save_dir = os.path.join(output_dir, model_name, "images")
        plt.savefig(os.path.join(save_dir, "accuracies.png"))
        plt.close()
    return acc_means, acc_sums



def get_md_experiment(all_configs, main_folder, search="remaining"):
    all_folders = main_folder
    elements_list = []
    
    if search == None:
        return all_configs
    
    if search == "remaining":
        for folder_name in all_folders:
            elements_list_match = re.search(r'\[.*?\]', folder_name)
            if elements_list_match:
                elements_list_str = elements_list_match.group(0)  # Extract the matched string
                elements_list.append(eval(elements_list_str))  # Convert the string to a list

    remaining_configs = []
    for config in all_configs:
        if config not in elements_list:
            remaining_configs.append(config)

    all_configs = remaining_configs[::-1]
    if search == "remaining":
        print(f"Running for remaining {len(all_configs)} MD configs:")
    else:
        if search is not None:
            print("Running for:")
    print(all_configs)
    
    return all_configs



def metadata_overview(all_metadata, all_titles, norm=True, normtype="minmax"):
    
    plt.figure(figsize=(7, 7))

    if norm == True:
        for i, data in enumerate(all_metadata):
            bins = 50
            if normtype == "minmax":
                norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            elif normtype == "z_score":
                norm_data = (data - np.mean(data)) / (np.std(data))

            plt.subplot(3, 3, i+1); plt.hist(norm_data, bins=bins); 
            plt.title(f"{all_titles[i]}")
            
        plt.suptitle(f"Overview over metadata distributions\n{normtype} normalized")
    else:
        for i, data in enumerate(all_metadata):
            bins = 50

            plt.subplot(3, 3, i+1); plt.hist(data, bins=bins)
            plt.title(all_titles[i])
        plt.suptitle("Overview over metadata distributions\nNot normalized")
    
    plt.tight_layout()
    plt.show()

    

def truth_distributions(all_metadata):
    normalized_metadata_minmax = []
    normalized_metadata_zscore = []
    normalized_metadata_l1 = []
    normalized_metadata_l2 = []

    for i, data in enumerate(all_metadata):
        # Min-Max Normalization
        norm_data_minmax = (data - np.min(data)) / (np.max(data) - np.min(data))
        normalized_metadata_minmax.append(norm_data_minmax)
        
        # Z-score Normalization (Standardization)
        norm_data_zscore = (data - np.mean(data)) / (np.std(data))
        normalized_metadata_zscore.append(norm_data_zscore)
        
        # L1 Normalization
        norm_data_l1 = data / np.sum(np.abs(data))
        normalized_metadata_l1.append(norm_data_l1)
        
        # L2 Normalization (Unit Vector Normalization)
        norm_data_l2 = data / np.linalg.norm(data)
        normalized_metadata_l2.append(norm_data_l2)


    random_cos_sims_minmax = []
    random_cos_sims_zscore = []
    random_cos_sims_l1 = []
    random_cos_sims_l2 = []

    for i in range(len(normalized_metadata_minmax[0])):
        i1 = np.random.randint(len(normalized_metadata_minmax[0]))
        i2 = np.random.randint(len(normalized_metadata_minmax[0]))

        # Cosine Similarity for Min-Max Normalization
        md1 = np.array(normalized_metadata_minmax)[:, i1]
        md2 = np.array(normalized_metadata_minmax)[:, i2]
        truth_minmax = metrics.pairwise.cosine_similarity((md1, md2))[0][1]
        random_cos_sims_minmax.append(truth_minmax)

        # Cosine Similarity for Z-score Normalization
        md1 = np.array(normalized_metadata_zscore)[:, i1]
        md2 = np.array(normalized_metadata_zscore)[:, i2]
        truth_zscore = metrics.pairwise.cosine_similarity((md1, md2))[0][1]
        random_cos_sims_zscore.append(truth_zscore)
        
        # Cosine Similarity for L1 Normalization
        md1 = np.array(normalized_metadata_l1)[:, i1]
        md2 = np.array(normalized_metadata_l1)[:, i2]
        truth_l1 = metrics.pairwise.cosine_similarity((md1, md2))[0][1]
        random_cos_sims_l1.append(truth_l1)
        
        # Cosine Similarity for L2 Normalization
        md1 = np.array(normalized_metadata_l2)[:, i1]
        md2 = np.array(normalized_metadata_l2)[:, i2]
        truth_l2 = metrics.pairwise.cosine_similarity((md1, md2))[0][1]
        random_cos_sims_l2.append(truth_l2)
        

    plt.figure(figsize=(7,7))
    plt.subplot(2, 2, 1)
    plt.hist(random_cos_sims_zscore, bins=100)
    plt.xlabel("Ground truth")
    plt.title("Standard normalized")

    plt.subplot(2, 2, 2)
    plt.hist(random_cos_sims_minmax, bins=100)
    plt.xlabel("Ground truth")
    plt.title("Min/max normalized")

    plt.subplot(2, 2, 3)
    plt.hist(random_cos_sims_l1, bins=100)
    plt.xlabel("Ground truth")
    plt.title("L1 normalization")

    plt.subplot(2, 2, 4)
    plt.hist(random_cos_sims_l2, bins=100)
    plt.xlabel("Ground truth")
    plt.title("L2 normalization (Unit vector)")

    plt.suptitle("Distribution of ground truths\naccording to cosine similarity")
    plt.tight_layout()
    #plt.savefig("Cossim_distribution.png")




#########################################
############ EXPERIMENTATION ############
#########################################


def calculate_accuracy(experiment, truths, predictions, lower_limit=0.9, upper_limit=1.0, bins=10, plot=True):
    
    top_truths = truths[np.where((predictions >= 0.9) & (predictions <= 1.0))]
    top_predictions = predictions[np.where((predictions >= 0.9) & (predictions <= 1.0))]
    
    truths = truths[np.where((predictions >= lower_limit) & (predictions <= upper_limit))]
    predictions = predictions[np.where((predictions >= lower_limit) & (predictions <= upper_limit))]
    
    errors = np.linspace(0, 1.0, bins+1)

    if not np.any((np.array(np.round(truths * 100) / 100) != 1.) & (np.array(np.round(truths * 100) / 100) != -1.)):
        #print("Only 1s and -1s for truths")
        return np.zeros(len(errors)), np.zeros(len(errors))

 
    def scoring(errors, truths, predictions):
        acc_sums = []
        acc_means = []
        for i, error in enumerate(errors):
            true_scores = []
            for t, p in zip(truths, predictions):
                if i == 0 and abs(t-p) < error:
                    score = 1
                elif abs(t-p) < error and abs(t-p) >= errors[i-1]:
                    score = 1
                else:
                    score = 0
                true_scores.append(score)
            acc_mean = np.sum(true_scores) / len(true_scores)  
            acc_means.append(acc_mean)
            acc_sums.append(np.sum(true_scores))
        return acc_means, acc_sums

    acc_means, acc_sums = scoring(errors, truths, predictions)

    if plot == True:

        plt.figure(figsize=(8,5))

        # plt.subplot(2,1,1)
        bars1 = plt.bar(np.linspace(-0.1, 0.9, bins+1), acc_sums, width=1/bins, align="edge", label="Number of correct guesses within error")
        for i, rect in enumerate(bars1):
            height = rect.get_height()
            if height != 0:
                total_sum = np.sum(acc_sums)
                percentage = height / total_sum * 100
                plt.text(rect.get_x() + rect.get_width() / 2, height, f'{percentage:.2f}%',
                        ha='center', va='bottom')
                
        #plt.bar(np.linspace(-0.1, 0.9, bins+1), list(accumulate(acc_sums)), width=0.1, align="edge", alpha=0.2, label="Accumulated")
        if not lower_limit == 0.9 and upper_limit == 1.0:
            top_acc_means, top_acc_sums = scoring(errors, top_truths, top_predictions)
            top_total_sum = np.sum(top_acc_sums)
            bars2 = plt.bar(np.linspace(-0.1, 0.9, bins+1), top_acc_sums, width=0.1, align="edge", alpha=0.5, label="Top predictions above 0.9")
            for i, rect in enumerate(bars2):
                height = rect.get_height()
                if height != 0:
                    percentage = height / top_total_sum * 100
                    if percentage >= 1.0:
                        plt.text(rect.get_x() + rect.get_width() / 2, height, f'{percentage:.2f}%',
                                ha='center', va='bottom', color='white')
        plt.legend()
        plt.xlim(-0.05, 1.05)
        plt.xlabel("Error")
        plt.ylabel("Counts")
        plt.title(f"Model accuracy as function of error for\npredictions in limit [{lower_limit}:{upper_limit}]\n\nExperiment: {experiment}")
        
    return acc_means, acc_sums


def get_accuracies(experiment_folders, sorter, lower_limit=0.9, upper_limit=1.0, bins=10, reverse=False, prints=False):
    
    accuracies = []
    experiments = []
    for experiment_folder in experiment_folders:

        for experiment in os.listdir(experiment_folder):
            folder_path = os.path.join(experiment_folder, experiment)
            
            if "batch_data.h5" in os.listdir(folder_path):
                h5_file = os.path.join(folder_path, "batch_data.h5")
            
                truths, predictions = get_batch_data_hdf5(h5_file, ["truths", "predictions"])
                accuracy, _ = calculate_accuracy(experiment, truths, predictions, lower_limit=lower_limit, upper_limit=upper_limit, bins=bins, plot=False)
                if np.min(accuracy) == 0 and np.max(accuracy) == 0:
                    print(f"Only 1s and -1s for truths for experiment:\n{experiment}")
                accuracies.append(accuracy)  
                experiments.append(experiment)  
            else:
                if prints:
                    print(f"No batch data for experiment {experiment}") 

    if sorter is not None:
        sorted_indices = np.argsort(sorter)
        if reverse:
            sorted_indices = np.argsort(sorter)[::-1]
        accuracies = np.array([accuracies[i] for i in sorted_indices])
        experiments = np.array([experiments[i] for i in sorted_indices])
    else:
        accuracies = np.array(accuracies)#[accuracies[i] for i in accuracies])
        experiments = np.array(experiments)
    
    return accuracies, experiments


def get_MSEs(experiment_folders, sorter, lower_limit=0.9, upper_limit=1.0, bins=10, reverse=False, prints=False):
    
    MSEs = []
    experiments = []
    for experiment_folder in experiment_folders:

        for experiment in os.listdir(experiment_folder):
            folder_path = os.path.join(experiment_folder, experiment)
            
            if "batch_data.h5" in os.listdir(folder_path):
                h5_file = os.path.join(folder_path, "batch_data.h5")
            
                truths, predictions = get_batch_data_hdf5(h5_file, ["truths", "predictions"])
                truths = truths[np.where((predictions >= lower_limit) & (predictions <= upper_limit))]
                predictions = predictions[np.where((predictions >= lower_limit) & (predictions <= upper_limit))]
                print(len(predictions))
                MSE = metrics.mean_squared_error(truths, predictions)
                if np.min(MSE) == 0 and np.max(MSE) == 0:
                    print(f"Only 1s and -1s for truths for experiment:\n{experiment}")
                MSEs.append(MSE)  
                experiments.append(experiment)  
            else:
                if prints:
                    print(f"No batch data for experiment {experiment}") 

    if sorter is not None:
        sorted_indices = np.argsort(sorter)
        if reverse:
            sorted_indices = np.argsort(sorter)[::-1]
        MSEs = np.array([MSEs[i] for i in sorted_indices])
        experiments = np.array([experiments[i] for i in sorted_indices])
    else:
        MSEs = np.array(MSEs)#[accuracies[i] for i in accuracies])
        experiments = np.array(experiments)
    
    return MSEs, experiments

def get_training_stats(experiment_folders=["/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs/_experiments/"],
                       prints=False):
    val_mse = []
    validation_loss = []
    training_loss = []
    train_mse = []
    model_name = []
    first_train_loss = []
    first_validation_loss = []
    second_train_loss = []
    second_validation_loss = []
    epochs = []
    experiments = []
    for experiment_folder in experiment_folders:
        for experiment in os.listdir(experiment_folder):
            folder_path = os.path.join(experiment_folder, experiment)

            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                if "training_stats.h5" in files and "batch_data.h5" in files:
                    h5_file = os.path.join(folder_path, "training_stats.h5")
                    model_name.append(experiment)
                    with h5py.File(h5_file, "r") as hf:
                        val_mse.append(hf["all_batches"]["validation_mse"][-1])
                        train_mse.append(hf["all_batches"]["train_mse"][-1])
                        validation_loss.append(hf["all_batches"]["validation_loss"][:])
                        training_loss.append(hf["all_batches"]["training_loss"][:])
                        first_train_loss.append(hf["all_batches"]["first_train_loss"][:])
                        first_validation_loss.append(hf["all_batches"]["first_val_loss"][:])
                        second_train_loss.append(hf["all_batches"]["second_train_loss"][:])
                        second_validation_loss.append(hf["all_batches"]["second_val_loss"][:])
                        epochs.append(hf["all_batches"]["epoch"][:])
                    experiments.append(experiment)
                else:
                    if prints:
                        print(f"No batch data for experiment {experiment}") 

    return experiments, val_mse, validation_loss, training_loss, train_mse, model_name, first_train_loss, \
        first_validation_loss, second_train_loss, second_validation_loss, epochs



def sort_values(sorter, val_mse, train_mse, model_name, validation_loss, training_loss,
                first_train_loss, first_validation_loss, second_train_loss, second_validation_loss,
                epochs, accuracies, reverse=False):
    sorted_indices = np.argsort(sorter)
    if reverse:
        sorted_indices = np.argsort(sorter)[::-1]
    
    val_mses = np.array(val_mse)[sorted_indices]
    train_mses = np.array(train_mse)[sorted_indices]
    experiments = np.array(model_name)[sorted_indices]

    validation_losses = [validation_loss[i] for i in sorted_indices]
    training_losses = [training_loss[i] for i in sorted_indices]
    first_train_losses = [first_train_loss[i] for i in sorted_indices]
    first_validation_losses = [first_validation_loss[i] for i in sorted_indices]
    second_train_losses = [second_train_loss[i] for i in sorted_indices]
    second_validation_losses = [second_validation_loss[i] for i in sorted_indices]
    sort_epochs = [epochs[i] for i in sorted_indices]
    sort_accuracies = np.array([accuracies[i] for i in sorted_indices])

    print(f"Variables sorted")
    return val_mses, train_mses, experiments, validation_losses, training_losses, first_train_losses, \
        first_validation_losses, second_train_losses, second_validation_losses, sort_epochs, sort_accuracies


def plot_experiments(experiments, variable, var_name, sorter="MSE", n=None, 
                     lower_limit=0.9, upper_limit=1.0, save_file=None, 
                     label_type="homemade", labels=None, x=None, x_label=None):
    if not n:
        n = len(variable)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, n))[::-1]

    all_labels = []
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(n):
        label = None
        if i < 300:
            if label_type == "md":
                elements_list_match = re.search(r'\[.*?\]', experiments[i])
                if elements_list_match:
                    elements_list_str = elements_list_match.group(0)  # Extract the matched string
                    elements_list = eval(elements_list_str)
                    label = str(elements_list) + experiments[i][elements_list_match.end():]
                    if "he" in experiments[i]:
                        label = "HI_" + label
                    
                    # Check if the elements include "batchnorm" and modify the label accordingly
                    if "batchnorm" in experiments[i]:
                        label = "BN_" + label

                    if "ELU" in experiments[i]:
                        label = "ELU_" + label

                    if "Leaky" in experiments[i]:
                        label = "LeakyReLU" + label

                    if "fcs" in experiments[i]:
                        label = "fcs" + label
                    
                    if not label.startswith("SN"):
                        label = "SN_" + label
            elif label_type == "homemade":
                label = labels[i]
            elif label_type == None:
                label=None
            else:
                label = experiments[i]
        if x == None:
            ax.scatter(i, variable[i], label=label, color=colors[i])
            ax.set_xlabel("Ranked model nr.")
        else:
            ax.scatter(x[i], variable[i], label=label, color=colors[i])

    all_labels.append(label)
    ax.set_ylabel(var_name)
    ax.set_xlabel(x_label)
    ax.set_title(f"{var_name} for {n} best models ranked by {sorter}")

    if var_name == "accuracy":
        ax.set_title(f"{var_name} for {n} best models for predictions within [{lower_limit:.2f}:{upper_limit:.2f}]")
        ax.set_ylim(-0.05, 1.05)
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1., 1))

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return all_labels



def get_remaining_experiments(all_short_titles, experiment_folder):
    

    
    all_title_configs = []
    for r in range(1, len(all_short_titles) + 1):
        all_title_configs.extend(itertools.combinations(all_short_titles, r))
    all_title_configs = [list(config) for config in all_title_configs]

    all_folders = os.listdir(experiment_folder)

    short_names = []
    for folder_name in all_folders:
        elements_list_match = re.search(r'\[.*?\]', folder_name)
        if elements_list_match:
            elements_list_str = elements_list_match.group(0)  # Extract the matched string
            short_names.append(eval(elements_list_str))  # Convert the string to a list

    remaining_configs = []
    for config in all_title_configs:
        if config not in short_names:
            remaining_configs.append(config)
    print("Number of remaining configs:", len(remaining_configs))
    #remaining_configs[::-1]
    return remaining_configs


def save_accuracy_images(experiment_folder, sorter_name, sorter, experiments, errors=None):

    if errors is None or len(errors) == 0:
        errors = np.linspace(-1.0, 1.0, 21)

    save_dir = "/lustre/astro/antonmol/learning_stuff/siamese_networks/notebooks/images/accuracies"
    save_folder = os.path.join(save_dir, os.path.basename(os.path.normpath(experiment_folder)))
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for i, error in enumerate(errors):
        if i == len(errors)-1:
            break
        lower_limit = error
        upper_limit = errors[i+1]
        accuracies = get_accuracies(experiment_folder, lower_limit=lower_limit, upper_limit=upper_limit, bins=10)
        sorted_indices = np.argsort(sorter)
        if sorter_name == "accuracy":
            sorted_indices = np.argsort(accuracies[:, 1])[::-1]
        accuracies = np.array([accuracies[i] for i in sorted_indices])
        save_file = os.path.join(save_folder, f"[{lower_limit:.1f}:{upper_limit:.1f}].png")
        plot_experiments(experiments, accuracies[:,1], "Accuracy", sorter=sorter_name, lower_limit=lower_limit, upper_limit=upper_limit,
                        save_file = save_file)

        print(f"Saved image at {save_file}")


