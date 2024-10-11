---
layout: post
title: UNet_Paper_Review
subtitle: Convolutional Networks for Biomedical Image Segmentation
excerpt_image: /assets/images/post_data/UNet_Paper_Review_banner.png
author: Kim Seojin
content_types: Paper_Review
research_areas: computer_vision
sidebar: []
paper_link: https://arxiv.org/abs/1505.04597
show_on_home: false
---

# UNet

# U-Net Paper Review

> **U-Net: Convolutional Networks for Biomedical Image Segmentation**
> 
> [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
> 

## Introduction

In the case of medical data, training datasets are often small.

U-Net demonstrates accurate image segmentation even with very few training examples.

The network is trained using the sliding-window approach.

- Sliding-window:

    An algorithm where a fixed-size window moves over the data, solving the problem using the data within the window.
    
    In other words, a window of a certain size moves across the image, detecting objects within the window.
    
    ![images_cha-suyeon_post_826faf40-bf3b-4bec-b823-5ef02dbec51c_image.png](/assets/images/post_data/UNet_Paper_Review_image/images_cha-suyeon_post_826faf40-bf3b-4bec-b823-5ef02dbec51c_image.png)
    
    If the object is much larger than the window, only part of the object enters the window, which may make it difficult to recognize the object. To address this, different shapes of windows can be used, or the input image size can be adjusted while keeping the window shape fixed.
    
    ![sliding-window-animated-adrian.gif](/assets/images/post_data/UNet_Paper_Review_image/sliding-window-animated-adrian.gif)
    
1. This network allows for localization.
2. Patch-based training data is more abundant than the number of training images.

Drawbacks:

1. Patch-based computation → slower processing $\downarrow$ → increased number of overlapping patches $\uparrow$ → increased number of redundant predictions $\uparrow$
2. Increased patches $\uparrow$ → requires more max-pooling operations $\uparrow$ → decreased localization accuracy $\downarrow$
3. Decreased patches $\downarrow$ → reduced feature extraction (a trade-off occurs)

![image.png](/assets/images/post_data/UNet_Paper_Review_image/image.png)

Even with a small number of training images, the performance improves $\uparrow$.

Features are extracted, and a low-resolution feature map is up-sampled and connected to the previously obtained feature map.

Up-sampling ↔ pooling → increased data volume $\uparrow$.

- Key characteristics of U-Net:
    1. During up-sampling, the number of feature channels increases $\uparrow$ → enabling context information to propagate to higher-resolution layers.
    2. The expansive path and contracting path are almost symmetrical, forming a U-shape.
    3. Fully convolutional layers are used instead of fully connected (FC) layers.
        
        FC layers use only information from patches for classification.
        
        Due to GPU memory limitations, there is a resolution constraint.
        
    4. When training data is limited, elastic deformation is applied for data augmentation.
    5. For cell segmentation, where objects of the same class are often adjacent, a weighted loss is used to separate them.

## Network Architecture

### Contracting Path

The architecture follows a typical convolutional network structure.

![img1.daumcdn.png](/assets/images/post_data/UNet_Paper_Review_image/img1.daumcdn.png)

![img1.daumcdn 1.png](/assets/images/post_data/UNet_Paper_Review_image/img1.daumcdn%201.png)

Two $3\times 3$ convolutions (unpadded convolutions) are used, followed by ReLU activation and $2\times 2$ max pooling (down-sampling).

→ The number of channels is doubled $\times 2$.

### Expansive Path

Up-sampling is applied.

![img1.daumcdn 2.png](/assets/images/post_data/UNet_Paper_Review_image/img1.daumcdn%202.png)

$2 \times 2$ convolution → the number of channels is halved $\times \frac{1}{2}$.

The contracting path and expanding path are connected for localization purposes.

## Training

### Loss function

$$
E = \sum_{x \in \Omega} \log(p_{l(x)}(x))
$$

### **Softmax**

$$
p_k(x) = \frac{exp(a_k(x))}{\Sigma_{k' = 1}^{K} exp(a_{k'}(x))}
$$

$$
w(x) = w_c(x) + w_0 \cdot exp\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)
$$

Elastic deformation is recognized as a key concept when using datasets with low deformation.

## Experiments

### Table 1

![image 1.png](/assets/images/post_data/UNet_Paper_Review_image/image%201.png)

### Table 2

![image 2.png](/assets/images/post_data/UNet_Paper_Review_image/image%202.png)

In Table 1, the EM segmentation challenge ranking is sorted by warping error. I wondered if it was appropriate to rank based on warping error, given that the U-Net proposed in this paper was introduced as a suitable model for biomedical applications.

## Conclusion

U-Net demonstrated excellent performance in biomedical segmentation.

Thanks to data augmentation using elastic deformations, the model could be trained with a small number of images within a reasonable time.
