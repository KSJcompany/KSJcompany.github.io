---
layout: post
title: R-CNN_Paper_Review
subtitle: Rich feature hierarchies for accurate object detection and semantic segmentation
excerpt_image: /assets/images/post_data/RCNN_Paper_Review_banner.png
author: Kim Seojin
content_types: Paper_Review
research_areas: computer_vision
sidebar: []
paper_link: https://arxiv.org/abs/1311.2524
show_on_home: false
---

# R-CNN

# R-CNN Paper Review

> **Rich feature hierarchies for accurate object detection and semantic segmentation**
> 
> [Rich feature hierarchies for accurate object detection and...](https://arxiv.org/abs/1311.2524)
> 

Object detection: Localization (determining the location of objects) + Classification (categorizing the detected objects).

→ 1-stage Detector: Solves localization (L) and classification (C) simultaneously (→ speed $\uparrow$) → accuracy $\downarrow$

![img1.daumcdn.png](/assets/images/post_data/RCNN_Paper_Review_image/img1.daumcdn.png)

→ 2-stage Detector: Solves localization (L) first, followed by classification (C) sequentially (→ speed $\downarrow$) → accuracy $\uparrow$ → R-CNN

![img1.daumcdn 1.png](/assets/images/post_data/RCNN_Paper_Review_image/img1.daumcdn%201.png)

## How R-CNN Works

![img1.daumcdn 2.png](/assets/images/post_data/RCNN_Paper_Review_image/img1.daumcdn%202.png)

1. Input the image.
2. Use the Selective Search algorithm to extract 2000 region proposals where objects are likely present.
3. Warp the extracted regions to a size of 227 × 227.
4. Input the warped region proposals into a fine-tuned AlexNet (a CNN pre-trained on ImageNet) to extract feature vectors.
5. Input the extracted feature vectors into a linear SVM and a bounding box regressor model to obtain confidence scores and adjusted bounding box coordinates.
6. Apply Non-Maximum Suppression (NMS) to output the optimal bounding box.

## R-CNN Structure

### Region Proposal

Find regions where objects are likely present.

(In the past, the sliding window method was used, but it was inefficient.)

![img1.daumcdn 3.png](/assets/images/post_data/RCNN_Paper_Review_image/img1.daumcdn%203.png)

**Selective Search** is widely used in segmentation.

It identifies objects based on differences in color, texture, and whether the object is surrounded by other objects.

→ Forms bounding boxes (creates random bounding boxes in multiple shapes) → progressively merges them → identifies objects.

### CNN

Once the region proposals are warped to 227 × 227, they are input into the CNN model.

→ Extracts a 4096-dimensional feature vector, creating a fixed-length feature vector.

### SVM

Used for classification.

SVM was chosen over softmax because when SVM was used, the mAP value was 54.2%, whereas with softmax, it was 50.9%.

- **Bounding Box Regression**
    
    ![img1.daumcdn 4.png](/assets/images/post_data/RCNN_Paper_Review_image/img1.daumcdn%204.png)
    
    A model that adjusts the bounding box generated by Selective Search to more accurately enclose the object.
    
    $$
    \hat G_x = P_w d_x(P) + P_x
    $$
    
    $$
    \hat G_y = P_h d_y(P) + P_y
    $$
    
    $$
    \hat G_w = P_w \exp(d_w(P))
    $$
    
    $$
    \hat G_h = P_h \exp(d_h(P))
    $$
    
    $$
    t_x = \frac{G_x - P_x}{P_w}
    $$
    
    $$
    t_y = \frac{G_y - P_y}{P_h}
    $$
    
    $$
    t_w = \log\left(\frac{G_w}{P_w}\right)
    $$
    
    $$
    t_h = \log\left(\frac{G_h}{P_h}\right)
    $$
    
    In the equation on the left, $\hat G$ is a variable that gets as close as possible to the ground truth (denoted as $G$ on the right).
    
    By substituting $\hat G$ for $G$ and $d$ for $t$ in the equation on the left, we obtain the equation on the right.
    

$$
w_* = \argmin_{\hat w_*} \sum^{N}_{i} (t_*^i - \hat w_*^T \phi_5(P^i))^2 + \|\hat w_*\|
$$

$d_*(P) = w^T_* \phi_5(P)$

The model learns by reducing the loss (the difference between $t$ and $d$).

## Results

![image.png](/assets/images/post_data/RCNN_Paper_Review_image/image.png)

It performs significantly better than previous models.

In the case of R-CNN BB, bounding box regression is applied.

## Drawbacks

1. Takes longer than previous models.
2. More complex than previous models.
3. Backpropagation is not possible (→ CNN cannot be updated).

These drawbacks are addressed, and performance is further improved with Fast R-CNN and Faster R-CNN.
