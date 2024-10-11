---
layout: post
title: ResNet_Paper_Review
subtitle: Deep Residual Learning for Image Recognition
author: Kim Seojin
content_types: Paper_Review
research_areas: computer_vision
sidebar: []
paper_link: https://arxiv.org/abs/1512.03385
show_on_home: false
---

# ResNet

# Review of the ResNet Paper

> Deep Residual Learning for Image Recognition  
> 
> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Abstract

This paper presents a **residual learning framework** to ease the training of deeper neural networks.

**Residual learning** refers to learning the difference (or residual) between the input and output, rather than learning the output directly.

Datasets used: ImageNet, CIFAR-10, and COCO object detection datasets.

Residual networks make optimization easier and achieve higher accuracy as the depth of the network increases.

## Introduction

Recent studies have shown that the depth of the network plays a crucial role in its accuracy. In particular, deeper models perform well on the **ImageNet** dataset.

> **Is learning better networks as easy as stacking more layers?**

### Problem 1: Convergence Problem

Deeper networks face the **vanishing/exploding gradients** problem, which hampers convergence from the very beginning.

**Solution**: Use **normalized initialization** and **intermediate normalization layers**.

This allows the model to converge even with dozens of layers using **stochastic gradient descent (SGD)** with backpropagation.

### Problem 2: Degradation Problem

![Degradation Problem](/assets/images/post_data/ResNet_Paper_Review_image/img.png)

As the network depth increases, accuracy saturates and then degrades. This issue is **not due to overfitting**, as both the **test error** and **training error** increase with deeper networks.

**Solution**: Use a **deep residual learning framework**.

In traditional networks, the underlying mapping is done as \( H(x) \).

In the residual network, they instead map \( F(x) = H(x) - x \), which makes optimization easier. The original mapping is then \( F(x) + x \).

Using **shortcut connections**, identity mapping is performed by adding the input directly to the output of the stacked layers. This does not introduce any additional parameters or computational complexity.

### Key Finding from ImageNet Dataset

- **Deep residual networks** are easier to optimize, whereas "plain" deep networks suffer from higher training errors as the depth increases.
- Residual networks enjoy accuracy gains with increased depth and outperform previous architectures.

Similar results are observed on the CIFAR-10 dataset.

## Deep Residual Learning

### Residual Learning

- **Identity mapping**: Passing the input directly without modification, where \( F(x) = 0 \).
- **Zero mapping**: Learning the mapping through nonlinear layers. While zero mapping is prone to optimization difficulties, identity mapping allows better tracking of errors.

If the added layers perform identity mapping, deeper models should not have a higher training error than shallower ones.

### Identity Mapping by Shortcuts

$$
y = F(x, \{ W_i \}) + x
$$

Where:
- \( x \): Input vector
- \( y \): Output vector
- \( F(x, \{ W_i \}) \): Residual mapping

The addition \( F + x \) is performed via shortcut connections using element-wise addition.

This formula holds when \( x \) and \( F \) have the same dimensions. If they differ:

$$
y = F(x, \{ W_i \}) + W_s x
$$

Here, \( W_s \) is used to adjust the dimensions.

### Network Architectures

![ResNet Architecture](/assets/images/post_data/ResNet_Paper_Review_image/ResNet_Fig.3(removed).png)

- **Plain Network**: Uses VGGNet as the baseline. Most convolutional layers use 3x3 filters.
- **Residual Network**: Adds shortcut connections to the plain network, using identity mapping without additional parameters.

For dimension matching, **projection shortcuts** are used where necessary.

### Implementation

- Image resize: 224x224
- **Batch normalization (BN)** is applied
- **SGD** with mini-batch size of 256
- Learning rate: 0.1
- Weight decay: 0.0001
- Momentum: 0.9
- Dropout is not used

## Experiments

### ImageNet Classification

![ImageNet Results](/assets/images/post_data/ResNet_Paper_Review_image/994C50365CB8A11F1D.jpg)

In the figure above, deeper **plain networks** (34-layer) show worse validation error than shallower networks (18-layer), demonstrating the **degradation problem**.

**Residual networks**, on the other hand, significantly reduce the **training error** and achieve better accuracy, resolving the degradation issue.

**34-layer ResNet** reduces top-1 error by 3.5% compared to its plain counterpart.

**Faster convergence** and easier optimization are also observed with ResNet.

### Identity vs. Projection Shortcuts

![Shortcuts Comparison](/assets/images/post_data/ResNet_Paper_Review_image/993D0B365CB8A11F0A.jpg)

- **A**: Zero-padding shortcuts (parameter-free).
- **B**: Projection shortcuts for dimension matching, with identity elsewhere.
- **C**: All shortcuts use projection.

**Performance**: C > B > A > Plain.

The differences among A, B, and C are not significant, indicating that projection shortcuts are not essential for resolving the degradation problem.

### CIFAR-10 and Analysis

![CIFAR-10 Results](/assets/images/post_data/ResNet_Paper_Review_image/99ED3B465CB8A11F09.jpg)

For the **CIFAR-10** dataset, which uses 32x32 images, similar trends are observed. Interestingly, deeper networks also show higher training errors, warranting further investigation.
