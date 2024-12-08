# Image similarity pipeline with the use of Siamese Neural Networks. 
This work is the backbone for a pipeline for identifying similar images using a Siamese Neural Network (SNN) - a complex deep neural network that is specialized to identify and map out features of images in order to learn to find similar images and those that are not.

The pipeline aims to analyze and extract significant features from images of protostellar systems (young stars and their surroundings). Its objective is to identify images within a vast dataset of simulated images that closely resemble a specified observational image. These are termed "best matches". Subsequently, it maps the distributions of system parameters, such as the mass of the star or the age of the system, for these best matches. These distributions could serve as a guide or an estimation of the physical parameters (mass, age, ...) of the observed system. 

Given that we do not have access to categorized data of protostellar systems, we instead provide each pair of training images with a measure of similarity computed from the system properties for each image. This is categorized as Weakly Supervised Metric Learning (WSML). The goal of this part of the thesis is thus to create the backbone of such a pipeline and to investigate 1: if the method can correctly identify similar images of protostellar systems based on WSML, and 2: whether the distribution of system properties collected from the most similar images can provide any insight to the properties of the target observed protostellar system.

The pipeline we propose include the following steps:
1. Encoding the images to a latent image of lower dimensions, where each pixel is a representation of features in the original image. This will provide the similarity-measuring model with a better interpretation of the input image.
2. Train a Siamese Neural Network on a training set of simulated images, labeled by the system parameters defined in the simulation.
3. Measure similarities between one observed image and a large set of simulated images using the trained Siamese network. The network takes as input pairs of images and outputs a value of the similarity.
4. Find the simulated images that are most similar to the observed image and compute distributions of their properties such as disk size, mass, and accretion rate.

The end result of the pipeline is first and foremost a mapped set of simulated protostellar systems that is of high similarity with a given observed protostellar system. Second, the best-match distributions of system properties can provide insight to further analysis of specific protostellar system.

# The pipeline

1. Setting up PyTorch DataLoaders with pairs of images using data_setup.py. For each pair, the ground truth similarity is computed from the cosine similarity of the sets of metadata from each image.
2. Build the various SNN models with model_builder.py
3. Setting up main training and validation loops with engine.py
4. Setting up hyperparameter tuning with tuning.py
5. Train the model with train.py
6. Use the trained model to find best matches between datasets and a target image using matching.py


# The general model structure
The Siamese Neural Network implements a similarity learning by utilizing the image analysis and representation abilities of a Convolutional Neural Network. It outputs complex, compressed representations for two images simultanously, giving rise to the ability of learning to represent similar images with similar representations and vice versa. 

![image](SNN_structure.png)



### More info
Are you interested in learning more about this model? Feel free to contact me by mail to anton14mol at gmail.com.
