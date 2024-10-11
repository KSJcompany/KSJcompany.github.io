---
layout: post
title: DCGAN_Paper_Review
subtitle: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
author: Kim Seojin
content_types: Paper_Review
research_areas: computer_vision
sidebar: []
paper_link: https://arxiv.org/abs/1511.06434
show_on_home: false
---

# DCGAN

# Review of the DCGAN Paper

> **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**
> 
> 
> [Unsupervised Representation Learning with Deep Convolutional...](https://arxiv.org/abs/1511.06434)
> 

DCGAN: A GAN network based on CNNs

→ Demonstrates that CNNs can be powerfully used in unsupervised learning.

GAN: A method for image representation → However, earlier GAN models were unstable during training, sometimes producing odd outputs from the generator.

- Key differences between traditional GANs and DCGANs:
    
    → Proposes Deep Convolutional GANs (DCGANs) that enable stable training.
    
    → Uses a discriminator trained for image classification, showing competitive performance compared to other unsupervised algorithms.
    
    → Visualizes the GAN’s filters to demonstrate what objects are drawn by each filter.
    
    → Shows the characteristics of the generator’s input vectors, enabling easier and more varied semantic manipulations.
    

## Network Architecture

![img1.daumcdn.png](/assets/images/post_data/DCGAN_Paper_Review_image/img1.daumcdn.png)

1. Replaces spatial pooling functions with strided convolutions
    
    : CNN → trains its own downsampling / generator → learns its own upsampling
    
2. Removes fully connected hidden layers
3. Adds batch normalization
    
    : However, applying it to all layers may reduce performance, so it is not used in the generator output and discriminator input.
    
4. Uses ReLU in all layers of the generator except for the output layer, which uses Tanh, while the discriminator uses LeakyReLU.

## Results

The paper trained models on the LSUN, ImageNet-1k, and Faces datasets (the latter crawled independently).

- LSUN
    
    Trained for 1 epoch (Figure 2)
    
    ![images_wilko97_post_06e15d67-c0ba-4a8d-9ad5-b53ebf4041e0_image.png](/assets/images/post_data/DCGAN_Paper_Review_image/images_wilko97_post_06e15d67-c0ba-4a8d-9ad5-b53ebf4041e0_image.png)
    
    Trained for 5 epochs (Figure 3)
    
    ![images_wilko97_post_a466f2bb-458d-4802-a183-39fa36b90840_image.png](/assets/images/post_data/DCGAN_Paper_Review_image/images_wilko97_post_a466f2bb-458d-4802-a183-39fa36b90840_image.png)
    
    After 5 epochs, underfitting was observed.
    
- ImageNet-1k
    
    ![images_wilko97_post_ef81c309-3efb-4581-9c70-cde8361495bd_image.png](/assets/images/post_data/DCGAN_Paper_Review_image/images_wilko97_post_ef81c309-3efb-4581-9c70-cde8361495bd_image.png)
    
    As seen in the table (Table 1), even for supervised image segmentation, the model performed well. DCGAN, trained on ImageNet-1k, also outperformed other algorithms in classifying the CIFAR-10 dataset, which it was not trained on.
    
- Walking in the Latent Space
    
    The paper also experimented with the latent space.
    
    ![images_wilko97_post_9863ffcf-7424-418d-8d8c-13eca956184d_image.png](/assets/images/post_data/DCGAN_Paper_Review_image/images_wilko97_post_9863ffcf-7424-418d-8d8c-13eca956184d_image.png)
    
    By linearly interpolating random points in Z from the DCGAN model, the results were surprisingly smooth, as seen in the image above. This suggests the model has learned in line with its intended purpose rather than just memorizing patterns.
    
- Visualizing Discriminator Features
    
    ![img1.daumcdn.png](/assets/images/post_data/DCGAN_Paper_Review_image/img1.daumcdn%201.png)
    
    It’s possible to see which feature areas the discriminator activates.
    
    A significant difference is observed between the models on the left and right.
    
- Forgetting to Draw Certain Objects
    
    ![img1.daumcdn.png](/assets/images/post_data/DCGAN_Paper_Review_image/img1.daumcdn%202.png)
    
    In the second row, a filter that had learned windows was removed, and the result is an image where the windows are replaced by walls or darkened curtains, highlighting how other features filled in the space.
    
- Faces Dataset
    
    Image generated from a single Z vector
    
    ![img1.daumcdn.png](/assets/images/post_data/DCGAN_Paper_Review_image/img1.daumcdn%203.png)
    
    Image generated from three Z vectors
    
    ![img1.daumcdn.png](/assets/images/post_data/DCGAN_Paper_Review_image/img1.daumcdn%204.png)
    
    Image rotation
    
    ![img1.daumcdn.png](/assets/images/post_data/DCGAN_Paper_Review_image/img1.daumcdn%205.png)
    
    → The rotation of the image was executed more successfully than expected.
