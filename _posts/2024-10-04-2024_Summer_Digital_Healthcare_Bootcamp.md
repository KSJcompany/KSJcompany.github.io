---
layout: post
title: 2024 Summer Digital Healthcare Bootcamp
subtitle: The implementation of a CT-based DNN model for early dementia diagnosis
excerpt_image: /assets/images/post_data/2024_Summer_Digital_Healthcare_Bootcamp_banner.png
author: Kim Seojin
content_types: Project
research_areas: medical_ai
sidebar: []
ppt_link: https://docs.google.com/presentation/d/11aO3R73HzhVL4vxbOoWTMGiyHflKhgbq/edit#slide=id.g279b9673f7d_9_15
video_link: https://youtu.be/_I1TLRsFS1Q?si=jVHASAcyb93xpNMG
---

## 2024 Summer Digital Healthcare Bootcamp

During the 2024 Summer Digital Healthcare Bootcamp, I participated in a 5-day intensive program focused on AI applications in healthcare. Our team worked on a project that aimed to develop a DNN model for early-stage dementia diagnosis using CT scans. Specifically, we focused on preprocessing based on ventricular presence and analyzing performance based on various parameters in fully connected layers.

We sought to add an additional stage—**Very Mild**—to the traditional three-stage Alzheimer’s severity model (Mild, Moderate, Severe) to better detect early signs of the disease. The dataset we used for this project was based on a Kaggle notebook, titled [Omdena Alzheimer's Classification using CNN](https://www.kaggle.com/code/vencerlanz09/omdena-alzheimer-s-classification-using-cnn/notebook).

In our approach, we identified the presence of brain ventricles as a significant factor in diagnosing dementia, and thus, we performed preprocessing accordingly. To evaluate the model’s performance, we focused on the following key performance metrics:

1. **Accuracy rate**  
2. **Loss rate**  
3. **Convergence rate** as the number of epochs increased  
4. **Variance between training accuracy and validation accuracy**

We further adjusted the model through parameter tuning, focusing on:

1. **Number of nodes in the fully connected layers**  
2. **Activation functions**  
3. **Optimizers**

Throughout the process, we iterated on these variables to find the optimal model configuration. We also explored **transfer learning** techniques and compared our model’s performance with that of the **ResNet** architecture to understand how our approach measured up against a more widely recognized model.

At the end of the bootcamp, we presented our findings. One key piece of feedback we received was to consider the hippocampus' role in Alzheimer’s during data preprocessing. We also noted that certain classes had significantly fewer images in the training dataset, leading to lower accuracy for those specific categories, which we identified as an area for improvement in future iterations of the project.
