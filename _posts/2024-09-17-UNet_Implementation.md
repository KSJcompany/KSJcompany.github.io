---
layout: post
title: UNet_Implementation
subtitle: Convolutional Networks for Biomedical Image Segmentation
excerpt_image: /assets/images/post_data/UNet_banner.png
author: Kim Seojin
content_types: Implementation
research_areas: computer_vision
sidebar: []
paper_link: https://arxiv.org/abs/1505.04597
ppt_link: /assets/U-Net.pdf
---

# UNet Implementation

# Implementing UNet

![u-net-architecture.png](/assets/images/post_data/UNet_Implementation_image/u-net-architecture.png)

## Writing the Code

### Commonly Used Blocks

- ConvBlock
    
    ![img1.daumcdn.png](/assets/images/post_data/UNet_Implementation_image/img1.daumcdn.png)
    
    ```python
    class ConvBlock(nn.Module):
        
    
        def __init__(self, in_channels, out_channels, kernel_size, stride = (1,1), padding = 0):
            super().__init__()
            self.convblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True)
            )
    
        def forward(self, x):
            return self.convblock(x)
    ```
    
    in_channels : Number of channels in the input image
    
    out_channels : Number of channels in the output
    
    kernel_size : Size of the convolutional filter
    
    stride : The step size for the filter during the convolution operation
    
    padding : Method of handling image boundaries 
    
- Up_conv
    
    ```python
    class Up_Conv(nn.Module):
        
    
        def __init__(self, in_channels, out_channels, kernel_size = (2,2), stride = (2,2), padding = 0):
            super().__init__()
            self.up_conv = nn.Sequential(
                # nn.Upsample(scale_factor=kernel_size, mode='bilinear'),
                # nn.Conv2d(in_channels, out_channels, (1,1), (1,1), padding)
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
            )
    
        def forward(self, x):
            return self.up_conv(x)
    ```
    
    Used during upsampling
    
- Center_Crop
    
    ```python
    def Center_Crop(big_tensor, little_tensor_shape):
        cut_y, cut_x = (int((big_tensor.shape[2] - little_tensor_shape[2])/2), int((big_tensor.shape[3] - little_tensor_shape[3])/2))
        return big_tensor[:,:,cut_y: cut_y + little_tensor_shape[2], cut_x: cut_x + little_tensor_shape[3]]
    ```
    
    These blocks are used during upsampling to prevent mismatches in tensor sizes during the upsampling process when running the UNet model.

    

### UNet Implementation

```python
class UNet(nn.Module):
    def __init__(self): 
        super(UNet, self).__init__()
        # Using maxpooling to reduce the resolution by half
        self.maxpool = nn.MaxPool2d((2,2))
        self.drop = nn.Dropout2d(0.5)     

        self.conv1 = ConvBlock(1, 64, 3)
        self.conv2 = ConvBlock(64, 128, 3)
        self.conv3 = ConvBlock(128, 256, 3)
        self.conv4 = ConvBlock(256, 512, 3)  

        self.conv5 = ConvBlock(512, 1024, 3)
        
        self.up_conv1 = Up_Conv(1024, 512)
        
        self.conv6 = ConvBlock(512 + 512, 512, 3) 

        self.up_conv2 = Up_Conv(512, 256)
        
        self.conv7 = ConvBlock(256 + 256, 256, 3) 

        self.up_conv3 = Up_Conv(256, 128)
        
        self.conv8 = ConvBlock(128 + 128, 128, 3) 

        self.up_conv4 = Up_Conv(128, 64)
        
        self.conv9 = ConvBlock(64 + 64, 64, 3)
        
        self.conv10 = nn.Conv2d(64, 1, 1) 

    def forward(self, x):
		    # contracting path
        x_conv1 = self.conv1(x)
        x = self.maxpool(x_conv1)

        x_conv2 = self.conv2(x)
        x = self.maxpool(x_conv2)

        x_conv3 = self.conv3(x)
        x = self.maxpool(x_conv3)

        x = self.conv4(x)
        x_drop4 = self.drop(x)
        x = self.maxpool(x_drop4)
        
        x = self.conv5(x)
        x = self.drop(x)

				# Expanding Path
        x = self.up_conv1(x)
        x = torch.cat((Center_Crop(x_drop4, x.shape), x), dim=1)
        x = self.conv6(x)

        x = self.up_conv2(x)
        x = torch.cat((Center_Crop(x_conv3, x.shape), x), dim=1)
        x = self.conv7(x)
        
        x = self.up_conv3(x)
        x = torch.cat((Center_Crop(x_conv2, x.shape), x), dim=1)
        x = self.conv8(x)

        x = self.up_conv4(x)
        x = torch.cat((Center_Crop(x_conv1, x.shape), x), dim=1)
        x = self.conv9(x)        
        
        result = torch.sigmoid(self.conv10(x))

        return result
```

## Running the Model

The dataset used is from the ISBI 2012 EM Segmentation Challenge

### Case without a Validation Set

- Results

![download.png](/assets/images/post_data/UNet_Implementation_image/download.png)

**Prediction results for test image (image #8)**

![8.png](/assets/images/post_data/UNet_Implementation_image/8.png)

![prediction_8.png](/assets/images/post_data/UNet_Implementation_image/prediction_8.png)

When I first ran the model, the dataset was divided into only training and test sets, so I ran it without a validation set.

### Case with a Validation Set

- Results

![download 1.png](/assets/images/post_data/UNet_Implementation_image/download%201.png)

You can see that the loss is lower, and the IoU is higher compared to the case without a validation set.

- IoU(Intersection over Union)
    
    A metric used in object detection and segmentation tasks, IoU represents the overlap between the predicted bounding box and the ground truth. The value ranges between 0 and 1.
    

**Prediction results for test image (image #8)**

![8.png](/assets/images/post_data/UNet_Implementation_image/8.png)

![prediction_8 (1).png](/assets/images/post_data/UNet_Implementation_image/prediction_8_(1).png)

When comparing the test image predictions between cases with and without a validation set, it appears that the prediction with a validation set produces a cleaner output with better separation of the membrane.