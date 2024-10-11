---
layout: post
title: Mask R-CNN_Paper_Review
subtitle: Mask R-CNN
author: Kim Seojin
content_types: Paper_Review
research_areas: computer_vision
sidebar: []
paper_link: https://arxiv.org/abs/1703.06870
---

# Mask R-CNN

# Mask R-CNN Paper Review

> **Mask R-CNN**
> 
> 
> [Mask R-CNN](/https://arxiv.org/abs/1703.06870)
> 

Mask R-CNN: A model that adds a mask branch to Faster R-CNN to simultaneously handle classification, bounding box regression, and segmentation masks.

### Image Segmentation

1. Semantic Segmentation
    
    : When performing segmentation, objects of the same class are not distinguished or separated, but rather displayed as the same region or color [left image].
    
2. Instance Segmentation (Mask R-CNN)
    
    : Unlike semantic segmentation, even if objects belong to the same class, they are distinguished and displayed as different regions or colors [right image].
    
    In Mask R-CNN, a mask is applied within the ROI (Region of Interest) to distinguish between instances of each class.
    
    ![img1.daumcdn.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/img1.daumcdn.png)
    

### Faster R-CNN

1-stage: Region Proposal Network (RPN) → Proposes candidates for the bounding box of each object.

2-stage: Fast R-CNN → Apply RoIPool → Perform classification and bounding box regression for each proposed candidate.

![faster_r-cnn_figure2.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/faster_r-cnn_figure2.png)

### RoIAlign

- Proposed to solve the problem of RoIPool.
    
    RoIPool: Extracts a smaller feature map (e.g., 7×7) from each RoI.
    
    At this stage, the RoI is first quantized and then divided into predefined sizes for pooling operations. If the continuous coordinate $x$ is quantized by applying $[x/16]$, this quantization can cause misalignment between the RoI and the extracted features.
    
    While this misalignment may not have a significant impact on classification, it can greatly affect tasks that require pixel-level accuracy.
    

RoIAlign eliminates the harsh quantization in RoIPool.

![RoIAlign-vs-RoIPool-RoIAlign-give-better-forward-output-and-the-gradient-is-more.jpg](/assets/images/post_data/Mask_RCNN_Paper_Review_image/RoIAlign-vs-RoIPool-RoIAlign-give-better-forward-output-and-the-gradient-is-more.jpg)

### Network Architecture

- Backbone:
    
    ResNet with 50 or 101 layers, or ResNeXt
    
    Uses Feature Pyramid Network (FPN)
    
    → Extracts features from the image.
    
- Head:
    
    Faster R-CNN + Mask branch ← The structure depends on the backbone.
    
    Performs bounding box recognition and mask prediction.
    
    ![img.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/img.png)
    

### Implementation Details

- Training:
    
    $Loss_{mask}$ ← Defined by positive RoI (IoU with a ground-truth ≥ 0.5).
    
    Image size: 800 pixels.
    
    Mini-batch: 2 images per GPU (Each image contains N sampled RoIs / C4 backbone → N = 64, FPN → N = 512).
    
    GPUs: 8.
    
    Learning rate: 0.02.
    
    Weight decay: 0.0001.
    
    Momentum: 0.9.
    
    When using ResNeXt → 1 image per GPU for training, learning rate: 0.01.
    
- Inference:
    
    Proposals from C4 backbone → 300 / Proposals from FPN → 1000.
    
    After the box prediction branch operates, non-maximum suppression is applied.
    
    The mask branch predicts K masks for each RoI.
    
    → K: The number of classes predicted by the classification branch.
    

### Results

Performance is measured using COCO metrics, including $AP$ (averaged over IoU thresholds), $AP_{50}$, $AP_{75}$, and $AP_{S}$, $AP_{M}$, $AP_{L}$ (AP at different scales). (AP: mask IoU)

![result1.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/result1.png)

Looking at Table 1 above, we can see that Mask R-CNN significantly outperforms previous SOTA models.

![result2.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/result2.png)

![result3.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/result3.png)

The figure above (Figure 6) compares the results of the FCIS model with the Mask R-CNN model, while the table (Table 2) evaluates the impact of different elements of the Mask R-CNN model on performance.
(a) Backbone Architecture
(b) Multinomial vs. Independent Masks
(c) RoIAlign
(d) Results from adjusting the stride value in RoIAlign
(e) Mask Branch

![result4.png](/assets/images/post_data/Mask_RCNN_Paper_Review_image/result4.png)
