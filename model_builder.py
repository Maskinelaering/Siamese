import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torcheval.metrics import MeanSquaredError
import numpy as np
import tqdm as tqdm
from sklearn import metrics

import data_setup

"""
This script is for setting up the Siamese network. 

"""


##### -------------- Step 2: Define model --------------------

#distance_function = "euclid"

class SiameseNetwork(nn.Module):

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
        super(SiameseNetwork, self).__init__()

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
            nn.ReLU(),
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

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_he_init, self).__init__()

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
            nn.ReLU(),
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



class SiameseNetwork_he_init_batchnorm(nn.Module):

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
        super(SiameseNetwork_he_init_batchnorm, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=1, padding=1),
            nn.BatchNorm2d(l1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=1, padding=1),
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(l2, l3, 3, stride=1, padding=1),
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(l3, l4, 3, stride=1, padding=1),
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=1, padding=1),
            nn.BatchNorm2d(l5),
            nn.ReLU()
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

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_he_init_batchnorm_ELU, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=1, padding=1),
            nn.BatchNorm2d(l1),
            nn.ELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(l1, l2, 3, stride=1, padding=1),
            nn.BatchNorm2d(l2),
            nn.ELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(l2, l3, 3, stride=1, padding=1),
            nn.BatchNorm2d(l3),
            nn.ELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(l3, l4, 3, stride=1, padding=1),
            nn.BatchNorm2d(l4),
            nn.ELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(l4, l5, 3, stride=1, padding=1),
            nn.BatchNorm2d(l5),
            nn.ReLU()
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
    For testing changes to SiameseNetwork model
    
    Output size for conv2d layer determined by:
    H_out = [(H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride] +1

    Example: H_in=800, kernel_size=3, padding=1, stride=2, dilation=1.
    H_out = [(800 + 2 - 1*(3-1) - 1) / 2] +1 = [799/2] +1 = 400.5  -- NOTE: Aparently rounds down to 400.

    For padding=1, dilation=1 it can be reduced to:
    H_out = [(H_in + 2 - kernel_size) / stride] +1

    """

    def __init__(self, l1=8, l2=16, l3=32, l4=64, l5=128, fc_units1=1024, fc_units2=128):
        super(SiameseNetwork_tester, self).__init__()

        "The architecture of the Siamese Network"
        

        # N, 1, size, size

        # CNN layers
        size = data_setup.input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, l1, 3, stride=1, padding=1), # -> N, l1, size, size
            nn.BatchNorm2d(l1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l1, size/2, size/2
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=1, padding=1), # -> N, l2, size/2, size/2
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l2, size/4, size/4
            nn.ReLU(),
            nn.Conv2d(l2, l3, 3, stride=1, padding=1), # -> N, l3, size/4, size/4
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l3, size/8, size/8
            nn.ReLU(),
            nn.Conv2d(l3, l4, 3, stride=1, padding=1), # -> N, l4, size/4, size/4
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1), # -> N, l1, size/16, size/16
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=1, padding=1), # -> N, l5, size/16, size/16
            nn.BatchNorm2d(l5),
            nn.ReLU(),
            #nn.MaxPool2d(3, stride=2, padding=1), # -> N, l5, size/32, size/32
            #nn.ReLU(),
        )

        output_size = size/(16)
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





















