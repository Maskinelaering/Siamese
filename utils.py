import torch
import torch.nn as nn
#import torchsummary
from torchsummary import summary
import numpy as np
import pandas as pd
from pathlib import Path
import os
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
    print(f"[INFO] Saving model to: {output_dir_path}")
    model_savename = "{}/{}.pth".format(output_dir_path, model_name)  # Name of saved model
    torch.save(obj=model.state_dict(),
             f=model_savename)

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
    Used to import data stored as hdf5 file.
    Parameters imported are all from during training or testing.
    truths: The label of similarity for input pair of images
    predictions: The predicted similarity by the model
    output1s: latent feature map output from first input image
    output2s: latent feature map output from second input image
    img1: first input image
    img2: second input image
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
                    data[name] = np.concatenate([batch_group[name][:].ravel()])
                else:
                    data[name] = np.concatenate([batch_group[name][:]])

    df = pd.DataFrame(data)
    return df[param_names].apply(lambda x: x.ravel()).values.T


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

#     lists = {name: np.array([]) for name in param_names}
#     print(lists)
#     with h5py.File(h5_filename, "r") as hf:
#         for batch_id, batch_group in hf.items():
#             for name in param_names:
#                 try: 
#                     batch_group[name]
#                 except:
#                     print(f"Parameter {name} not in file {h5_filename}")

#                 if name in ["truths", "predictions"]:
#                     lists[name] = np.concatenate((lists[name], batch_group[name][:].ravel()))
#                 else:
#                     lists[name] = np.concatenate((lists[name], batch_group[name][:]))
                
#                 #lists[name] = np.concatenate(lists[name])

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

#     return lists #truths, predictions, output1s, output2s, img1s, img2s, md1s, md2s, md1s_raw, md2s_raw



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
    plt.legend()
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

    def running(image_number, target):
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
            save_name = os.path.join(vis_folder, f"attr_{method}_tar{target}_img{image_number}.png")
            summed = False
        elif method == "layer":
            grad = LayerIntegratedGradients(combined_forward, layer)
            summed = True
            save_name = os.path.join(vis_folder, f"attr_{method}_summed_img{image_number}.png")
        
        
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
            figure, axis = viz.visualize_image_attr(attrib, origin, method="blended_heat_map",sign="all",
                                    show_colorbar=True, title=f"Overlayed Integrated Gradients\nMethod: {method}")
            
            figure.savefig(save_name)
            print(f"Saved attributions figure at {save_name}")
            plt.close()

    # Running for each input image
    if method == "layer":
        running(0, targets)
        running(1, targets)
    else:
        for target in targets:
            target = int(target)
            running(0, target)
            running(1, target)



def animate_attr(output_dir, model_name, n, save=False):
    # Get a list of all image files in the directory containing "imgn" in their name
    image_dir = os.path.join(output_dir, model_name, "images", "visuals")
    image_files = glob.glob(os.path.join(image_dir, f"*img{n}*.png"))
    image_files.sort()

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
    with h5py.File(h5_filename, "r") as hf:
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
