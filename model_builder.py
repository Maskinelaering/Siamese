import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torcheval.metrics import MeanSquaredError
import numpy as np
import os
import tqdm as tqdm
from sklearn import metrics

import data_setup

"""
This script is for setting up the Siamese network. 

"""
def calc_divs(input_size=400, kernel_size=3, stride=1, padding=1, dilation=1):
    res = ((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    divs = input_size / res
    return divs, np.round(divs)

os.environ["CUDA_VISIBLE_DEVICES"]="3"  # Choose which device to use (astro01 has 4 gpu's)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### -------------- Step 2: Define model --------------------

#distance_function = "euclid"

class SiameseNetwork(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )
        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    # def set_parameters(self, **kwargs):
    #     """
    #     Set parameters of the SiameseNetwork model.
    #     """
    #     for key, value in kwargs.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else:
    #             print(f"Attribute '{key}' does not exist in the SiameseNetwork model.")
                
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction


class SiameseNetwork_ELU(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_ELU, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.ELU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.ELU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.ELU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ELU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.ELU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.ELU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ELU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )
        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ELU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    # def set_parameters(self, **kwargs):
    #     """
    #     Set parameters of the SiameseNetwork model.
    #     """
    #     for key, value in kwargs.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else:
    #             print(f"Attribute '{key}' does not exist in the SiameseNetwork model.")
                
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction
    
class SiameseNetwork_LeakyReLU(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_LeakyReLU, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.LeakyReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.LeakyReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.LeakyReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.LeakyReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.LeakyReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.LeakyReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.LeakyReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.LeakyReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.LeakyReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )
        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

                
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":
            prediction = F.cosine_similarity(output1, output2)

        elif data_setup.distance_function == "euclid":
            prediction = F.pairwise_distance(output1, output2)

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction


class SiameseNetwork_he_init(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_he_init, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        print("SIZE:", size)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )
        ls = [1, l1, l2, l3, l4, l5]
        n = 0
        new_size = size
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv2d):
                new_size = new_size // 2
                dummy_input = torch.randn(1, ls[n], size, size)
                output = layer(dummy_input)
                output_size = output.size()
                print(f"Output size of Conv2d layer {i}: {output_size}")
                n += 1

        # Compute the output size of the final Conv2d layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, size, size)
            final_conv_output = self.encoder(dummy_input)
            final_conv_output_size = final_conv_output.size()
            print("Output size of the final Conv2d layer:", final_conv_output_size)
        
        # Calculate the size of the flattened features
        output_height = final_conv_output_size[2]
        output_width = final_conv_output_size[3]
        fc_input_size = l5 * output_height * output_width
        print("FC1 input:", fc_input_size)

        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        print("OUTPUTSIZE:", output_size)
        print("L5:", l5)
        print("FC1 input size:", l5*int((output_size**2)))
        print("FC1 output size:", fc_units1)
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        # Apply Kaiming initialization to the weights of the layers
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction



class SiameseNetwork_he_init_batchnorm(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=400, kernel_size=3, padding=1, stride=1, dilation=1.
    H_out = [(400 + 2 - 1*(3-1) - 1) / 1] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_he_init_batchnorm, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1),
            nn.BatchNorm2d(l1),
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1),
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2),
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2),
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3),
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3),
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4),
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4),
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5),
            nn.BatchNorm2d(l5),
            nn.ReLU()
)


        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        # Apply Kaiming initialization to the weights of the layers
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction


class SiameseNetwork_batchnorm(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=400, kernel_size=3, padding=1, stride=1, dilation=1.
    H_out = [(400 + 2 - 1*(3-1) - 1) / 1] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_batchnorm, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1),
            nn.BatchNorm2d(l1),
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1),
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2),
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2),
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3),
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3),
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4),
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4),
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5),
            nn.BatchNorm2d(l5),
            nn.ReLU()
)


        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction



class SiameseNetwork_he_init_batchnorm_ELU(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_he_init_batchnorm_ELU, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1),
            nn.BatchNorm2d(l1),
            nn.ELU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1),
            nn.ELU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2),
            nn.BatchNorm2d(l2),
            nn.ELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2),
            nn.ELU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3),
            nn.BatchNorm2d(l3),
            nn.ELU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3),
            nn.ELU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4),
            nn.BatchNorm2d(l4),
            nn.ELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4),
            nn.ELU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5),
            nn.BatchNorm2d(l5),
            nn.ReLU()
)


        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        # Apply Kaiming initialization to the weights of the layers
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction




class SiameseNetwork_noPool(nn.Module):

    """
    Setup a Siamese Neural Network. 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Aparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_noPool, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, 800, 800

        # CNN layers
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=2, padding=1), # -> N, l1, 400, 400
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=2, padding=1), # -> N, l2, 200, 200
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(l2, l3, 3, stride=2, padding=1), # -> N, l3, 100, 100
            nn.ReLU(),
            nn.Conv2d(l3, l4, 7, stride=5, padding=1), # -> N, l4, 20, 20 
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=2, padding=1), # -> N, l5, 10, 10 
            nn.ReLU(),
        )

        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*10*10, fc_units1), #OBS: Update if image dimensions change..
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":
            prediction = F.cosine_similarity(output1, output2)

        #if data_setup.distance_function == "euclid":
            #prediction = metrics.euclidean_distances(output1, output2)
            #prediction = (output1 - output2).pow(2).sum(1).sqrt()

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction



class SiameseNetwork_Leung(nn.Module):

    """
    Setup a Siamese Neural Network. 

    
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_Leung, self).__init__()

        "The architecture of the Siamese Network"
        "Taken from Leung et al. https://arxiv.org/abs/1904.02906"

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=2, padding=1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=2, padding=1), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(l2, l3, 3, stride=2, padding=1), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, 7, stride=5, padding=1), # -> N, l4, size/40, size/40 
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=2, padding=1), # -> N, l5, size/80, size/80
            nn.ReLU(),
        )

        # Fully connected layers - convert to 1D array
        output_size = size/(80)
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), #OBS: Update if image dimensions change..
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":
            prediction = F.cosine_similarity(output1, output2)

        if data_setup.distance_function == "euclid":
            #prediction = metrics.euclidean_distances(output1, output2)
            prediction = (output1 - output2).pow(2).sum(-1).sqrt()

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction
    


class SiameseNetwork_tester(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, l6=64, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3, k6=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1, s6=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1, p6=1,
                 pk1=3, pk2=3, pk3=3, pk4=3, pk5=3,
                 ps1=2, ps2=2, ps3=2, ps4=2, ps5=2,
                 pp1=1, pp2=1, pp3=1, pp4=1, pp5=1):
        super(SiameseNetwork, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk5, stride=ps5, padding=pp5), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l5, l6, k6, stride=s6, padding=p6),
            nn.ReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )
        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction
    


class SiameseNetwork_alexnet(nn.Module):
    

    """
    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Aparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_alexnet, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 11, stride=4, padding=1), # -> N, l1, size, size
            nn.BatchNorm2d(l1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, 5, stride=1, padding=2), # -> N, l2, size/2, size/2
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, 3, stride=1, padding=0), # -> N, l3, size/4, size/4
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.Conv2d(l3, l4, 3, stride=1, padding=0), # -> N, l4, size/4, size/4
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(l4, l5, 3, stride=1, padding=0), # -> N, l5, size/16, size/16
            nn.BatchNorm2d(l5),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l2, size/4, size/4
            nn.ReLU(),
            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )

        output_size = 10 #size/(16)
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), #OBS: Update if image dimensions change..
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fc_units1),
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":
            prediction = F.cosine_similarity(output1, output2)

        if data_setup.distance_function == "euclid":
            prediction = (output1 - output2).pow(2).sum(-1).sqrt()

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction
    


class SiameseNetwork_sigmoid(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_sigmoid, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=1, padding=1), # -> N, l1, size, size
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=1, padding=1), # -> N, l2, size/2, size/2
            nn.Sigmoid(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, 3, stride=1, padding=1), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, 3, stride=1, padding=1), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=1, padding=1), # -> N, l5, size/16, size/16 
            nn.ReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )

        output_size = size/(16) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction




class SiameseAutoencoder(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseAutoencoder, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
        nn.ConvTranspose2d(l5, l4, 3, stride=2, padding=1, output_padding=1),  # -> N, l4, size/8, size/8
        nn.ReLU(),
        nn.ConvTranspose2d(l4, l3, 3, stride=2, padding=1, output_padding=1),  # -> N, l3, size/4, size/4
        nn.ReLU(),
        nn.ConvTranspose2d(l3, l2, 3, stride=2, padding=1, output_padding=1),  # -> N, l2, size/2, size/2
        nn.ReLU(),
        nn.ConvTranspose2d(l2, l1, 3, stride=2, padding=1, output_padding=1),  # -> N, l1, size, size
        nn.ReLU(),
        nn.ConvTranspose2d(l1, 1, 3, stride=1, padding=1),  # -> N, 1, size, size
        nn.Sigmoid()  # To ensure output is between 0 and 1
        )

        self.decoder2 = nn.Sequential(
        nn.ConvTranspose2d(l5, l4, 3, stride=2, padding=1, output_padding=1),  # -> N, l4, size/8, size/8
        nn.ReLU(),
        nn.ConvTranspose2d(l4, l3, 3, stride=2, padding=1, output_padding=1),  # -> N, l3, size/4, size/4
        nn.ReLU(),
        nn.ConvTranspose2d(l3, l2, 3, stride=2, padding=1, output_padding=1),  # -> N, l2, size/2, size/2
        nn.ReLU(),
        nn.ConvTranspose2d(l2, l1, 3, stride=2, padding=1, output_padding=1),  # -> N, l1, size, size
        nn.ReLU(),
        nn.ConvTranspose2d(l1, 1, 3, stride=1, padding=1),  # -> N, 1, size, size
        nn.Sigmoid()  # To ensure output is between 0 and 1
        )



        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        encoded = self.encoder(x)
        output = encoded.view(encoded.size()[0], -1)
        output = self.fc(output)
        return output, encoded

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1, encoded1 = self.forward_once(input1)
        output2, encoded2 = self.forward_once(input2)

        decoded1 = self.decoder1(encoded1)
        decoded2 = self.decoder2(encoded2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)


        prediction = prediction.to(self.device).float()
        return output1, output2, prediction, encoded1, encoded2, decoded1, decoded2




class SiameseNetwork_extended(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_extended, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size 512
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2 256
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4 128
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8 64
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16 32
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ReLU(),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l5, size/32, size/32 16
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(l5, l5, k5, stride=s5, padding=p5), # -> N, l5, size/32, size/32 
            nn.ReLU(),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l5, size/64, size/64 8
            nn.ReLU(),
            nn.Conv2d(l5, l5, k5, s5, p5), # -> N, l5, size/64, size/64
            nn.ReLU()

        )
        strides = [s1, s2, s3, s4, s5, s5, s5, ps1, ps2, ps3, ps4, ps4, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        self.fc = nn.Sequential(
            nn.Linear(l5*int((output_size**2)), fc_units1), 
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_units1, fc_units2),
            #nn.ReLU(inplace=True),
            #nn.Linear(2*l5,2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction





class SiameseNetwork_fcs(nn.Module):

    """
    Setup a Siamese Neural Network. 
    Inputs: 

    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2*1 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Apparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    Output size for maxpool2d layer determined by:
    H_out = [(H_in + 2*padding - kernel_size) / stride] +1

    Parameters:
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0
        padding_mode (str, optional) - 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    See more at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    
    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, 
                 fc_units1=1024, fc_units2=128,
                 fc_units3=None, fc_units4=None, fc_units5=None,
                 k1=3, k2=3, k3=3, k4=3, k5=3,
                 s1=1, s2=1, s3=1, s4=1, s5=1,
                 p1=1, p2=1, p3=1, p4=1, p5=1,
                 pk1=3, pk2=3, pk3=3, pk4=3,
                 ps1=2, ps2=2, ps3=2, ps4=2,
                 pp1=1, pp2=1, pp3=1, pp4=1):
        super(SiameseNetwork_fcs, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, k1, stride=s1, padding=p1), # -> N, l1, size, size
            nn.ReLU(),
            nn.MaxPool2d(pk1, stride=ps1, padding=pp1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, k2, stride=s2, padding=p2), # -> N, l2, size/2, size/2
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk2, stride=ps2, padding=pp2), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, k3, stride=s3, padding=p3), # -> N, l3, size/4, size/4
            nn.ReLU(),
            nn.MaxPool2d(pk3, stride=ps3, padding=pp3), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, k4, stride=s4, padding=p4), # -> N, l4, size/8, size/8 
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(pk4, stride=ps4, padding=pp4), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, k5, stride=s5, padding=p5), # -> N, l5, size/16, size/16 
            nn.ReLU(),

            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )
        strides = [s1, s2, s3, s4, s5, ps1, ps2, ps3, ps4]
        divis = np.prod([x for x in strides if x >= 2])
        output_size = size/(divis) #NOTE: Update if model dimensions change..
        # Fully connected layers - convert to 1D array
        
        if fc_units5 is not None:
            self.fc = nn.Sequential(
                nn.Linear(l5*int((output_size**2)), fc_units1), 
                nn.ReLU(inplace=True),
                nn.Linear(fc_units1, fc_units2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_units2, fc_units3),
                nn.ReLU(inplace=True),
                nn.Linear(fc_units3, fc_units4),
                nn.ReLU(inplace=True),
                nn.Linear(fc_units4, fc_units5),
            )
        elif fc_units4 is not None and fc_units5 is None:
            self.fc = nn.Sequential(
                nn.Linear(l5*int((output_size**2)), fc_units1), 
                nn.ReLU(inplace=True),
                nn.Linear(fc_units1, fc_units2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_units2, fc_units3),
            )
        elif fc_units3 is not None and fc_units4 is None:
            self.fc = nn.Sequential(
                nn.Linear(l5*int((output_size**2)), fc_units1), 
                nn.ReLU(inplace=True),
                nn.Linear(fc_units1, fc_units2),
                nn.ReLU(inplace=True),
                nn.Linear(fc_units2, fc_units3),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(l5*int((output_size**2)), fc_units1), 
                nn.ReLU(inplace=True),
                nn.Linear(fc_units1, fc_units2),
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        x = x.to(self.device)
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        input1 = input1.to(self.device).float()
        input2 = input2.to(self.device).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if data_setup.distance_function == "cosine":

            prediction = F.cosine_similarity(output1, output2)
            
            #prediction = (1 + prediction) / 2  # Mapping from [-1, 1] to [0, 1]
            #truth = (1 + truth) / 2  # Mapping from [-1, 1] to [0, 1]
            #prediction = torch.flatten(prediction, start_dim=1)

        elif data_setup.distance_function == "euclid":
            #prediction = (output1 - output2).pow(2).sum(-1).sqrt()
            prediction = F.pairwise_distance(output1, output2)
            #prediction = torch.sigmoid(-prediction)
            #prediction = prediction_tensor.numpy()
            #prediction = prediction.ravel()[0]

            #prediction = np.sqrt(np.sum((output1 - output2)**2))

        prediction = prediction.to(self.device).float()
        return output1, output2, prediction





