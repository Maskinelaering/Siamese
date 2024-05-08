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

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
import glob

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
    #/lustre/astro/antonmol/learning_stuff/autoencoder/encoder_modular/outputs/test_autoencoder_model
    output_dir_path = Path(model_dir_path)
    output_dir_path.mkdir(parents=True,
                         exist_ok=True)
    
    # Create model save path
    #assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"

    # Save the model state_dict()"
    
    model_savename = "{}/{}.pth".format(output_dir_path, model_name)  # Name of saved model
    torch.save(obj=model.state_dict(),
             f=model_savename)

    if structure_df is not None:
        print(f"[INFO] Saved model to: {output_dir_path}")
        # Save the model summary to a text file
        #result, params_info = torchsummary.summary_string(model, [(1,800,800),(1,800,800)], device="cuda")
        summary_filename = "{}/{}_summary.txt".format(output_dir_path, model_name)
        
        structure_df.to_csv(summary_filename, sep='\t', index=True, header=False)
        # with open(summary_filename, 'w') as summary_file:
        #     summary_file.write(structure_df)



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

# def get_batch_data_hdf5(h5_filename, param_names):
#     """
#     Used to import data stored as hdf5 file.
#     Parameters imported are all from during training or testing.
#     truths: The label of similarity for input pair of images
#     predictions: The predicted similarity by the model
#     output1s: latent feature map output from first input image
#     output2s: latent feature map output from second input image
#     img1: first input image
#     img2: second input image
#     """

#     data = {name: [] for name in param_names}

#     with h5py.File(h5_filename, "r") as hf:
#         for batch_id, batch_group in hf.items():
#             for name in param_names:
#                 try: 
#                     batch_group[name]
#                 except:
#                     print(f"Parameter {name} not in file {h5_filename}. \n Parameters: {batch_group.keys()}")
#                     break
#                 if name in ["truths", "predictions"]:
#                     data[name] = np.concatenate([batch_group[name][:].ravel()])
#                 else:
#                     data[name] = np.concatenate([batch_group[name][:]])
#                     print(f"{name}", data[name])


#     return np.column_stack([data[name] for name in param_names])
#     # df = pd.DataFrame(data)
    # return df[param_names].apply(lambda x: x.ravel()).values.T


# def get_batch_data_hdf5(h5_filename, param_names):
#     """
#     Used to import data stored as hdf5 file.
#     Parameters imported are all from during training or testing.
#     truths: The label of similarity for input pair of images
#     predictions: The predicted similarity by the model
#     output1s: latent feature map output from first input image
#     output2s: latent feature map output from second input image
#     img1: first input image
#     img2: second input image
#     """

#     # truths_list, predictions_list, output1s_list, output2s_list = [], [], [], []
#     # img1s_list, img2s_list = [], []
#     # md1_list, md2_list = [], []
#     # md1_list_raw, md2_list_raw = [], []
#     # with h5py.File(h5_filename, "r") as hf:
#     #     for batch_id, batch_group in hf.items():
#     #         truths_list.append(batch_group["truths"][:].ravel())
#     #         predictions_list.append(batch_group["predictions"][:].ravel())
#     #         output1s_list.append(batch_group["out1"][:])
#     #         output2s_list.append(batch_group["out2"][:])
#     #         img1s_list.append(batch_group["img1"][:])
#     #         img2s_list.append(batch_group["img2"][:])
#     #         md1_list.append(batch_group["metadata1"][:])
#     #         md2_list.append(batch_group["metadata2"][:])
#     #         md1_list_raw.append(batch_group["metadata1_raw"][:])
#     #         md2_list_raw.append(batch_group["metadata2_raw"][:])

#     # truths = np.concatenate(truths_list)
#     # predictions = np.concatenate(predictions_list)
#     # img1s = np.concatenate(img1s_list)
#     # img2s = np.concatenate(img2s_list)
#     # output1s = np.concatenate(output1s_list)
#     # output2s = np.concatenate(output2s_list)
#     # md1s = np.concatenate(md1_list)
#     # md2s = np.concatenate(md2_list)
#     # md1s_raw = np.concatenate(md1_list_raw)
#     # md2s_raw = np.concatenate(md2_list_raw)

#     return #truths, predictions, output1s, output2s, img1s, img2s, md1s, md2s, md1s_raw, md2s_raw



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


        # for batch_id, batch_group in hf.items():
        #     training_loss.append(batch_group["training_loss"][:])
        #     first_train_loss.append(batch_group["first_train_loss"][:])
        #     second_train_loss.append(batch_group["second_train_loss"][:])
        #     validation_loss.append(batch_group["validation_loss"][:])
        #     first_val_loss.append(batch_group["first_val_loss"][:])
        #     second_val_loss.append(batch_group["second_val_loss"][:])


    #epochs = np.arange(1, len(training_loss[0])+1)
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
    print(f"[INFO] Saved training evolution plot at {save_dir}/training_evol.png")

    plt.figure()
    plt.plot(epochs, train_mse, color="b", label="Train MSE")
    plt.plot(epochs, val_mse, color="r", label="Validation MSE")
    plt.legend()
    plt.title(f"Model: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(f"{save_dir}/MSEs.png")








def plot_attr(model, 
              img1s, 
              img2s, 
              output_dir,
              model_name,
              device,
              targets: list,
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
    print("\n[INFO] Model interpretation started.")
    
    print("\nComputing integrated gradients...")
    print(f"... on targets: {targets}")


    img1 = torch.tensor(img1s[0:1], requires_grad=True).to(device)
    img2 = torch.tensor(img2s[0:1], requires_grad=True).to(device)
    print(img1.shape)
    combined_images = torch.cat((img1, img2), dim=1)

    # baseline1 = torch.tensor([0.0]).to(device)
    # baseline2 = torch.tensor([0.0]).to(device)
    
    
    vis_folder = os.path.join(output_dir, model_name, "images", "attributions")
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)
    #save_dir = os.path.join(output_dir, model_name, "images", "attributions", "attr")
    

    checkpoint = torch.load(f"{output_dir}/{model_name}/{model_name}.pth")

    def running(image_number, target, sign):
        """
        This function runs the integrated gradients analysis for one of the input images
        and will run twice to analyse both images.
        """
        
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
        

        if np.isnan(attrib).any():
            print("WARNING: Attributions contains nan values")
            print("Using matplotlib instead (NOTE: image saving needs correction)")
            plt.figure()
            plt.imshow(origin, alpha=0.5)
            plt.imshow(attrib, alpha=0.5)
            plt.savefig(save_name)

            plt.figure()
            plt.imshow(attrib)
            plt.savefig(save_name)
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

    # def animate(i):
    #     update_image(i)

    ani = FuncAnimation(fig, animate, frames=len(image_files), interval=500)
    html_output = HTML(ani.to_jshtml())

    clear_output(wait=True)
    

    if save == True:
        save_name = os.path.join(image_dir, f'attr_vis_anim_img{n}.gif')
        ani.save(save_name, writer='pillow', dpi=300)

    display(html_output)





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
    plt.yscale('log')
    plt.title('Gradient Norms over Epochs')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    save_file = os.path.join(output_dir, model_name, "images", "gradient_evolution")
    plt.savefig(save_file, bbox_inches='tight')



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

    # print(f"{len(possible_combinations)} number of possible combinations for")
    # print("Kernel sizes: ",init_kernel_sizes)
    # print("Paddings: ",init_paddings)
    # print("Strides: ",init_strides)

    # print(f"\n{len(combs)} number of possible combinations for")
    # print("Kernel sizes: ",specified_kernel_sizes)
    # print("Paddings: ",specified_paddings)
    # print("Strides: ",specified_strides)


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
        
    return acc_means, acc_sums






def get_md_experiment(all_configs, search="remaining"):
    all_folders = os.listdir("/lustre/astro/antonmol/learning_stuff/siamese_networks/outputs/_experiments/metadata")
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

    print("remain", remaining_configs)
    all_configs = remaining_configs[::-1]
    if search == "remaining":
        print(f"Running for remaining {len(all_configs)} MD configs:")
    else:
        if search is not None:
            print("Running for:")
    print(all_configs)
    
    return all_configs