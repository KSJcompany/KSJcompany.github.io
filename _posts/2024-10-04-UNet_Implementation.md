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
    
    Epoch 1/300, Loss: 0.2335, IoU: 0.7711
    Epoch 2/300, Loss: 0.2305, IoU: 0.7711
    Epoch 3/300, Loss: 0.2274, IoU: 0.7711
    Epoch 4/300, Loss: 0.2236, IoU: 0.7711
    Epoch 5/300, Loss: 0.2187, IoU: 0.7711
    Epoch 6/300, Loss: 0.2121, IoU: 0.7711
    Epoch 7/300, Loss: 0.2020, IoU: 0.7711
    Epoch 8/300, Loss: 0.1833, IoU: 0.7711
    Epoch 9/300, Loss: 0.1718, IoU: 0.7711
    Epoch 10/300, Loss: 0.1698, IoU: 0.7711
    Epoch 11/300, Loss: 0.1679, IoU: 0.7711
    Epoch 12/300, Loss: 0.1667, IoU: 0.7711
    Epoch 13/300, Loss: 0.1657, IoU: 0.7711
    Epoch 14/300, Loss: 0.1652, IoU: 0.7711
    Epoch 15/300, Loss: 0.1645, IoU: 0.7711
    Epoch 16/300, Loss: 0.1636, IoU: 0.7711
    Epoch 17/300, Loss: 0.1627, IoU: 0.7711
    Epoch 18/300, Loss: 0.1618, IoU: 0.7711
    Epoch 19/300, Loss: 0.1609, IoU: 0.7711
    Epoch 20/300, Loss: 0.1599, IoU: 0.7711
    Epoch 21/300, Loss: 0.1589, IoU: 0.7711
    Epoch 22/300, Loss: 0.1578, IoU: 0.7711
    Epoch 23/300, Loss: 0.1566, IoU: 0.7711
    Epoch 24/300, Loss: 0.1553, IoU: 0.7711
    Epoch 25/300, Loss: 0.1539, IoU: 0.7711
    Epoch 26/300, Loss: 0.1523, IoU: 0.7711
    Epoch 27/300, Loss: 0.1503, IoU: 0.7711
    Epoch 28/300, Loss: 0.1477, IoU: 0.7711
    Epoch 29/300, Loss: 0.1444, IoU: 0.7712
    Epoch 30/300, Loss: 0.1400, IoU: 0.7715
    Epoch 31/300, Loss: 0.1344, IoU: 0.7761
    Epoch 32/300, Loss: 0.1284, IoU: 0.7857
    Epoch 33/300, Loss: 0.1233, IoU: 0.8005
    Epoch 34/300, Loss: 0.1185, IoU: 0.8087
    Epoch 35/300, Loss: 0.1144, IoU: 0.8127
    Epoch 36/300, Loss: 0.1128, IoU: 0.8115
    Epoch 37/300, Loss: 0.1119, IoU: 0.8117
    Epoch 38/300, Loss: 0.1083, IoU: 0.8180
    Epoch 39/300, Loss: 0.1074, IoU: 0.8187
    Epoch 40/300, Loss: 0.1075, IoU: 0.8185
    Epoch 41/300, Loss: 0.1079, IoU: 0.8174
    Epoch 42/300, Loss: 0.1088, IoU: 0.8168
    Epoch 43/300, Loss: 0.1052, IoU: 0.8234
    Epoch 44/300, Loss: 0.1040, IoU: 0.8244
    Epoch 45/300, Loss: 0.1041, IoU: 0.8243
    Epoch 46/300, Loss: 0.1056, IoU: 0.8220
    Epoch 47/300, Loss: 0.1057, IoU: 0.8237
    Epoch 48/300, Loss: 0.1024, IoU: 0.8287
    Epoch 49/300, Loss: 0.1020, IoU: 0.8285
    Epoch 50/300, Loss: 0.1033, IoU: 0.8263
    Epoch 51/300, Loss: 0.1040, IoU: 0.8266
    Epoch 52/300, Loss: 0.1014, IoU: 0.8311
    Epoch 53/300, Loss: 0.1007, IoU: 0.8315
    Epoch 54/300, Loss: 0.1015, IoU: 0.8301
    Epoch 55/300, Loss: 0.1025, IoU: 0.8293
    Epoch 56/300, Loss: 0.1010, IoU: 0.8324
    Epoch 57/300, Loss: 0.0997, IoU: 0.8339
    Epoch 58/300, Loss: 0.0998, IoU: 0.8333
    Epoch 59/300, Loss: 0.1007, IoU: 0.8321
    Epoch 60/300, Loss: 0.1005, IoU: 0.8332
    Epoch 61/300, Loss: 0.0991, IoU: 0.8355
    Epoch 62/300, Loss: 0.0986, IoU: 0.8357
    Epoch 63/300, Loss: 0.0991, IoU: 0.8349
    Epoch 64/300, Loss: 0.0995, IoU: 0.8347
    Epoch 65/300, Loss: 0.0987, IoU: 0.8365
    Epoch 66/300, Loss: 0.0977, IoU: 0.8378
    Epoch 67/300, Loss: 0.0976, IoU: 0.8377
    Epoch 68/300, Loss: 0.0980, IoU: 0.8372
    Epoch 69/300, Loss: 0.0980, IoU: 0.8376
    Epoch 70/300, Loss: 0.0972, IoU: 0.8392
    Epoch 71/300, Loss: 0.0965, IoU: 0.8401
    Epoch 72/300, Loss: 0.0963, IoU: 0.8401
    Epoch 73/300, Loss: 0.0967, IoU: 0.8397
    Epoch 74/300, Loss: 0.0966, IoU: 0.8403
    Epoch 75/300, Loss: 0.0956, IoU: 0.8419
    Epoch 76/300, Loss: 0.0949, IoU: 0.8428
    Epoch 77/300, Loss: 0.0947, IoU: 0.8430
    Epoch 78/300, Loss: 0.0950, IoU: 0.8427
    Epoch 79/300, Loss: 0.0950, IoU: 0.8429
    Epoch 80/300, Loss: 0.0941, IoU: 0.8447
    Epoch 81/300, Loss: 0.0932, IoU: 0.8459
    Epoch 82/300, Loss: 0.0927, IoU: 0.8465
    Epoch 83/300, Loss: 0.0927, IoU: 0.8463
    Epoch 84/300, Loss: 0.0939, IoU: 0.8444
    Epoch 85/300, Loss: 0.0939, IoU: 0.8457
    Epoch 86/300, Loss: 0.0925, IoU: 0.8476
    Epoch 87/300, Loss: 0.0911, IoU: 0.8492
    Epoch 88/300, Loss: 0.0903, IoU: 0.8501
    Epoch 89/300, Loss: 0.0903, IoU: 0.8500
    Epoch 90/300, Loss: 0.0915, IoU: 0.8480
    Epoch 91/300, Loss: 0.0917, IoU: 0.8489
    Epoch 92/300, Loss: 0.0903, IoU: 0.8512
    Epoch 93/300, Loss: 0.0899, IoU: 0.8503
    Epoch 94/300, Loss: 0.0882, IoU: 0.8533
    Epoch 95/300, Loss: 0.0885, IoU: 0.8531
    Epoch 96/300, Loss: 0.0885, IoU: 0.8536
    Epoch 97/300, Loss: 0.0890, IoU: 0.8524
    Epoch 98/300, Loss: 0.0886, IoU: 0.8525
    Epoch 99/300, Loss: 0.0869, IoU: 0.8551
    Epoch 100/300, Loss: 0.0863, IoU: 0.8568
    Epoch 101/300, Loss: 0.0861, IoU: 0.8575
    Epoch 102/300, Loss: 0.0921, IoU: 0.8470
    Epoch 103/300, Loss: 0.0907, IoU: 0.8504
    Epoch 104/300, Loss: 0.0871, IoU: 0.8533
    Epoch 105/300, Loss: 0.0863, IoU: 0.8572
    Epoch 106/300, Loss: 0.0871, IoU: 0.8574
    Epoch 107/300, Loss: 0.0879, IoU: 0.8532
    Epoch 108/300, Loss: 0.0843, IoU: 0.8590
    Epoch 109/300, Loss: 0.0854, IoU: 0.8590
    Epoch 110/300, Loss: 0.0858, IoU: 0.8586
    Epoch 111/300, Loss: 0.0857, IoU: 0.8566
    Epoch 112/300, Loss: 0.0843, IoU: 0.8592
    Epoch 113/300, Loss: 0.0836, IoU: 0.8610
    Epoch 114/300, Loss: 0.0844, IoU: 0.8595
    Epoch 115/300, Loss: 0.0837, IoU: 0.8599
    Epoch 116/300, Loss: 0.0821, IoU: 0.8626
    Epoch 117/300, Loss: 0.0832, IoU: 0.8614
    Epoch 118/300, Loss: 0.0833, IoU: 0.8602
    Epoch 119/300, Loss: 0.0813, IoU: 0.8638
    Epoch 120/300, Loss: 0.0829, IoU: 0.8623
    Epoch 121/300, Loss: 0.0823, IoU: 0.8617
    Epoch 122/300, Loss: 0.0813, IoU: 0.8639
    Epoch 123/300, Loss: 0.0821, IoU: 0.8633
    Epoch 124/300, Loss: 0.0821, IoU: 0.8620
    Epoch 125/300, Loss: 0.0811, IoU: 0.8641
    Epoch 126/300, Loss: 0.0813, IoU: 0.8644
    Epoch 127/300, Loss: 0.0821, IoU: 0.8627
    Epoch 128/300, Loss: 0.0805, IoU: 0.8645
    Epoch 129/300, Loss: 0.0808, IoU: 0.8647
    Epoch 130/300, Loss: 0.0802, IoU: 0.8657
    Epoch 131/300, Loss: 0.0820, IoU: 0.8629
    Epoch 132/300, Loss: 0.0809, IoU: 0.8637
    Epoch 133/300, Loss: 0.0816, IoU: 0.8625
    Epoch 134/300, Loss: 0.0794, IoU: 0.8671
    Epoch 135/300, Loss: 0.0822, IoU: 0.8629
    Epoch 136/300, Loss: 0.0802, IoU: 0.8644
    Epoch 137/300, Loss: 0.0798, IoU: 0.8661
    Epoch 138/300, Loss: 0.0801, IoU: 0.8663
    Epoch 139/300, Loss: 0.0796, IoU: 0.8659
    Epoch 140/300, Loss: 0.0792, IoU: 0.8670
    Epoch 141/300, Loss: 0.0786, IoU: 0.8678
    Epoch 142/300, Loss: 0.0797, IoU: 0.8663
    Epoch 143/300, Loss: 0.0795, IoU: 0.8653
    Epoch 144/300, Loss: 0.0781, IoU: 0.8686
    Epoch 145/300, Loss: 0.0792, IoU: 0.8663
    Epoch 146/300, Loss: 0.0778, IoU: 0.8685
    Epoch 147/300, Loss: 0.0784, IoU: 0.8681
    Epoch 148/300, Loss: 0.0795, IoU: 0.8658
    Epoch 149/300, Loss: 0.0812, IoU: 0.8640
    Epoch 150/300, Loss: 0.0810, IoU: 0.8617
    Epoch 151/300, Loss: 0.0780, IoU: 0.8695
    Epoch 152/300, Loss: 0.0818, IoU: 0.8630
    Epoch 153/300, Loss: 0.0776, IoU: 0.8682
    Epoch 154/300, Loss: 0.0777, IoU: 0.8701
    Epoch 155/300, Loss: 0.0782, IoU: 0.8679
    Epoch 156/300, Loss: 0.0763, IoU: 0.8715
    Epoch 157/300, Loss: 0.0781, IoU: 0.8688
    Epoch 158/300, Loss: 0.0766, IoU: 0.8708
    Epoch 159/300, Loss: 0.0782, IoU: 0.8678
    Epoch 160/300, Loss: 0.0757, IoU: 0.8726
    Epoch 161/300, Loss: 0.0787, IoU: 0.8675
    Epoch 162/300, Loss: 0.0756, IoU: 0.8714
    Epoch 163/300, Loss: 0.0764, IoU: 0.8711
    Epoch 164/300, Loss: 0.0767, IoU: 0.8708
    Epoch 165/300, Loss: 0.0766, IoU: 0.8699
    Epoch 166/300, Loss: 0.0750, IoU: 0.8733
    Epoch 167/300, Loss: 0.0758, IoU: 0.8723
    Epoch 168/300, Loss: 0.0754, IoU: 0.8725
    Epoch 169/300, Loss: 0.0757, IoU: 0.8715
    Epoch 170/300, Loss: 0.0772, IoU: 0.8704
    Epoch 171/300, Loss: 0.0790, IoU: 0.8671
    Epoch 172/300, Loss: 0.0797, IoU: 0.8643
    Epoch 173/300, Loss: 0.0768, IoU: 0.8705
    Epoch 174/300, Loss: 0.0783, IoU: 0.8695
    Epoch 175/300, Loss: 0.0773, IoU: 0.8687
    Epoch 176/300, Loss: 0.0757, IoU: 0.8728
    Epoch 177/300, Loss: 0.0766, IoU: 0.8711
    Epoch 178/300, Loss: 0.0753, IoU: 0.8726
    Epoch 179/300, Loss: 0.0748, IoU: 0.8737
    Epoch 180/300, Loss: 0.0757, IoU: 0.8722
    Epoch 181/300, Loss: 0.0743, IoU: 0.8742
    Epoch 182/300, Loss: 0.0749, IoU: 0.8734
    Epoch 183/300, Loss: 0.0742, IoU: 0.8744
    Epoch 184/300, Loss: 0.0742, IoU: 0.8742
    Epoch 185/300, Loss: 0.0738, IoU: 0.8751
    Epoch 186/300, Loss: 0.0749, IoU: 0.8732
    Epoch 187/300, Loss: 0.0736, IoU: 0.8749
    Epoch 188/300, Loss: 0.0733, IoU: 0.8759
    Epoch 189/300, Loss: 0.0743, IoU: 0.8745
    Epoch 190/300, Loss: 0.0739, IoU: 0.8745
    Epoch 191/300, Loss: 0.0727, IoU: 0.8765
    Epoch 192/300, Loss: 0.0729, IoU: 0.8768
    Epoch 193/300, Loss: 0.0741, IoU: 0.8745
    Epoch 194/300, Loss: 0.0724, IoU: 0.8768
    Epoch 195/300, Loss: 0.0729, IoU: 0.8765
    Epoch 196/300, Loss: 0.0724, IoU: 0.8773
    Epoch 197/300, Loss: 0.0729, IoU: 0.8767
    Epoch 198/300, Loss: 0.0730, IoU: 0.8756
    Epoch 199/300, Loss: 0.0716, IoU: 0.8786
    Epoch 200/300, Loss: 0.0737, IoU: 0.8756
    Epoch 201/300, Loss: 0.0730, IoU: 0.8758
    Epoch 202/300, Loss: 0.0712, IoU: 0.8790
    Epoch 203/300, Loss: 0.0714, IoU: 0.8796
    Epoch 204/300, Loss: 0.0740, IoU: 0.8750
    Epoch 205/300, Loss: 0.0723, IoU: 0.8766
    Epoch 206/300, Loss: 0.0703, IoU: 0.8808
    Epoch 207/300, Loss: 0.0723, IoU: 0.8783
    Epoch 208/300, Loss: 0.0726, IoU: 0.8764
    Epoch 209/300, Loss: 0.0729, IoU: 0.8763
    Epoch 210/300, Loss: 0.0708, IoU: 0.8799
    Epoch 211/300, Loss: 0.0736, IoU: 0.8761
    Epoch 212/300, Loss: 0.0721, IoU: 0.8771
    Epoch 213/300, Loss: 0.0715, IoU: 0.8786
    Epoch 214/300, Loss: 0.0703, IoU: 0.8810
    Epoch 215/300, Loss: 0.0716, IoU: 0.8788
    Epoch 216/300, Loss: 0.0714, IoU: 0.8785
    Epoch 217/300, Loss: 0.0699, IoU: 0.8811
    Epoch 218/300, Loss: 0.0696, IoU: 0.8824
    Epoch 219/300, Loss: 0.0712, IoU: 0.8794
    Epoch 220/300, Loss: 0.0715, IoU: 0.8785
    Epoch 221/300, Loss: 0.0700, IoU: 0.8806
    Epoch 222/300, Loss: 0.0689, IoU: 0.8835
    Epoch 223/300, Loss: 0.0700, IoU: 0.8814
    Epoch 224/300, Loss: 0.0699, IoU: 0.8810
    Epoch 225/300, Loss: 0.0697, IoU: 0.8818
    Epoch 226/300, Loss: 0.0683, IoU: 0.8837
    Epoch 227/300, Loss: 0.0687, IoU: 0.8835
    Epoch 228/300, Loss: 0.0692, IoU: 0.8827
    Epoch 229/300, Loss: 0.0695, IoU: 0.8820
    Epoch 230/300, Loss: 0.0687, IoU: 0.8835
    Epoch 231/300, Loss: 0.0697, IoU: 0.8820
    Epoch 232/300, Loss: 0.0727, IoU: 0.8770
    Epoch 233/300, Loss: 0.0708, IoU: 0.8788
    Epoch 234/300, Loss: 0.0686, IoU: 0.8833
    Epoch 235/300, Loss: 0.0682, IoU: 0.8852
    Epoch 236/300, Loss: 0.0698, IoU: 0.8815
    Epoch 237/300, Loss: 0.0682, IoU: 0.8842
    Epoch 238/300, Loss: 0.0671, IoU: 0.8862
    Epoch 239/300, Loss: 0.0671, IoU: 0.8862
    Epoch 240/300, Loss: 0.0675, IoU: 0.8856
    Epoch 241/300, Loss: 0.0666, IoU: 0.8867
    Epoch 242/300, Loss: 0.0662, IoU: 0.8875
    Epoch 243/300, Loss: 0.0655, IoU: 0.8888
    Epoch 244/300, Loss: 0.0657, IoU: 0.8887
    Epoch 245/300, Loss: 0.0673, IoU: 0.8859
    Epoch 246/300, Loss: 0.0700, IoU: 0.8817
    Epoch 247/300, Loss: 0.0720, IoU: 0.8773
    Epoch 248/300, Loss: 0.0671, IoU: 0.8852
    Epoch 249/300, Loss: 0.0683, IoU: 0.8856
    Epoch 250/300, Loss: 0.0693, IoU: 0.8824
    Epoch 251/300, Loss: 0.0673, IoU: 0.8851
    Epoch 252/300, Loss: 0.0667, IoU: 0.8875
    Epoch 253/300, Loss: 0.0669, IoU: 0.8862
    Epoch 254/300, Loss: 0.0652, IoU: 0.8893
    Epoch 255/300, Loss: 0.0657, IoU: 0.8892
    Epoch 256/300, Loss: 0.0661, IoU: 0.8877
    Epoch 257/300, Loss: 0.0648, IoU: 0.8901
    Epoch 258/300, Loss: 0.0647, IoU: 0.8905
    Epoch 259/300, Loss: 0.0652, IoU: 0.8893
    Epoch 260/300, Loss: 0.0653, IoU: 0.8893
    Epoch 261/300, Loss: 0.0635, IoU: 0.8918
    Epoch 262/300, Loss: 0.0635, IoU: 0.8926
    Epoch 263/300, Loss: 0.0636, IoU: 0.8922
    Epoch 264/300, Loss: 0.0644, IoU: 0.8909
    Epoch 265/300, Loss: 0.0644, IoU: 0.8905
    Epoch 266/300, Loss: 0.0629, IoU: 0.8928
    Epoch 267/300, Loss: 0.0636, IoU: 0.8921
    Epoch 268/300, Loss: 0.0631, IoU: 0.8931
    Epoch 269/300, Loss: 0.0650, IoU: 0.8901
    Epoch 270/300, Loss: 0.0643, IoU: 0.8902
    Epoch 271/300, Loss: 0.0624, IoU: 0.8939
    Epoch 272/300, Loss: 0.0614, IoU: 0.8954
    Epoch 273/300, Loss: 0.0616, IoU: 0.8957
    Epoch 274/300, Loss: 0.0618, IoU: 0.8949
    Epoch 275/300, Loss: 0.0622, IoU: 0.8942
    Epoch 276/300, Loss: 0.0615, IoU: 0.8951
    Epoch 277/300, Loss: 0.0620, IoU: 0.8947
    Epoch 278/300, Loss: 0.0608, IoU: 0.8968
    Epoch 279/300, Loss: 0.0602, IoU: 0.8972
    Epoch 280/300, Loss: 0.0610, IoU: 0.8964
    Epoch 281/300, Loss: 0.0620, IoU: 0.8947
    Epoch 282/300, Loss: 0.0632, IoU: 0.8927
    Epoch 283/300, Loss: 0.0627, IoU: 0.8930
    Epoch 284/300, Loss: 0.0616, IoU: 0.8952
    Epoch 285/300, Loss: 0.0594, IoU: 0.8985
    Epoch 286/300, Loss: 0.0598, IoU: 0.8984
    Epoch 287/300, Loss: 0.0596, IoU: 0.8987
    Epoch 288/300, Loss: 0.0591, IoU: 0.8992
    Epoch 289/300, Loss: 0.0581, IoU: 0.9010
    Epoch 290/300, Loss: 0.0576, IoU: 0.9020
    Epoch 291/300, Loss: 0.0575, IoU: 0.9019
    Epoch 292/300, Loss: 0.0567, IoU: 0.9032
    Epoch 293/300, Loss: 0.0581, IoU: 0.9004
    Epoch 294/300, Loss: 0.0581, IoU: 0.9018
    Epoch 295/300, Loss: 0.0605, IoU: 0.8969
    Epoch 296/300, Loss: 0.0611, IoU: 0.8953
    Epoch 297/300, Loss: 0.0585, IoU: 0.8996
    Epoch 298/300, Loss: 0.0588, IoU: 0.8995
    Epoch 299/300, Loss: 0.0580, IoU: 0.9016
    Epoch 300/300, Loss: 0.0578, IoU: 0.9019
    Saved PyTorch Model State to model.pth
    

![download.png](/assets/images/post_data/UNet_Implementation_image/download.png)

**Prediction results for test image (image #8)**

![8.png](/assets/images/post_data/UNet_Implementation_image/8.png)

![prediction_8.png](/assets/images/post_data/UNet_Implementation_image/prediction_8.png)

When I first ran the model, the dataset was divided into only training and test sets, so I ran it without a validation set.

### Case with a Validation Set

- Results
    
    Epoch 1/300 -> Train Loss: 0.2534, Train IoU: 0.0000, Val Loss: 0.2514, Val IoU: 0.0000
    Epoch 2/300 -> Train Loss: 0.2503, Train IoU: 0.2715, Val Loss: 0.2480, Val IoU: 0.7832
    Epoch 3/300 -> Train Loss: 0.2467, Train IoU: 0.7797, Val Loss: 0.2438, Val IoU: 0.7742
    Epoch 4/300 -> Train Loss: 0.2421, Train IoU: 0.7711, Val Loss: 0.2386, Val IoU: 0.7742
    Epoch 5/300 -> Train Loss: 0.2365, Train IoU: 0.7711, Val Loss: 0.2318, Val IoU: 0.7742
    Epoch 6/300 -> Train Loss: 0.2289, Train IoU: 0.7711, Val Loss: 0.2223, Val IoU: 0.7742
    Epoch 7/300 -> Train Loss: 0.2181, Train IoU: 0.7711, Val Loss: 0.2078, Val IoU: 0.7742
    Epoch 8/300 -> Train Loss: 0.2001, Train IoU: 0.7711, Val Loss: 0.1812, Val IoU: 0.7742
    Epoch 9/300 -> Train Loss: 0.1737, Train IoU: 0.7711, Val Loss: 0.1763, Val IoU: 0.7742
    Epoch 10/300 -> Train Loss: 0.1729, Train IoU: 0.7711, Val Loss: 0.1650, Val IoU: 0.7742
    Epoch 11/300 -> Train Loss: 0.1663, Train IoU: 0.7711, Val Loss: 0.1646, Val IoU: 0.7742
    Epoch 12/300 -> Train Loss: 0.1650, Train IoU: 0.7711, Val Loss: 0.1629, Val IoU: 0.7742
    Epoch 13/300 -> Train Loss: 0.1633, Train IoU: 0.7711, Val Loss: 0.1618, Val IoU: 0.7742
    Epoch 14/300 -> Train Loss: 0.1625, Train IoU: 0.7711, Val Loss: 0.1612, Val IoU: 0.7742
    Epoch 15/300 -> Train Loss: 0.1618, Train IoU: 0.7711, Val Loss: 0.1601, Val IoU: 0.7742
    Epoch 16/300 -> Train Loss: 0.1606, Train IoU: 0.7711, Val Loss: 0.1589, Val IoU: 0.7742
    Epoch 17/300 -> Train Loss: 0.1595, Train IoU: 0.7711, Val Loss: 0.1579, Val IoU: 0.7742
    Epoch 18/300 -> Train Loss: 0.1584, Train IoU: 0.7711, Val Loss: 0.1568, Val IoU: 0.7742
    Epoch 19/300 -> Train Loss: 0.1571, Train IoU: 0.7711, Val Loss: 0.1556, Val IoU: 0.7742
    Epoch 20/300 -> Train Loss: 0.1558, Train IoU: 0.7711, Val Loss: 0.1544, Val IoU: 0.7742
    Epoch 21/300 -> Train Loss: 0.1545, Train IoU: 0.7711, Val Loss: 0.1530, Val IoU: 0.7742
    Epoch 22/300 -> Train Loss: 0.1530, Train IoU: 0.7711, Val Loss: 0.1514, Val IoU: 0.7742
    Epoch 23/300 -> Train Loss: 0.1513, Train IoU: 0.7711, Val Loss: 0.1496, Val IoU: 0.7742
    Epoch 24/300 -> Train Loss: 0.1493, Train IoU: 0.7712, Val Loss: 0.1476, Val IoU: 0.7744
    Epoch 25/300 -> Train Loss: 0.1471, Train IoU: 0.7713, Val Loss: 0.1453, Val IoU: 0.7747
    Epoch 26/300 -> Train Loss: 0.1445, Train IoU: 0.7715, Val Loss: 0.1424, Val IoU: 0.7752
    Epoch 27/300 -> Train Loss: 0.1415, Train IoU: 0.7721, Val Loss: 0.1390, Val IoU: 0.7766
    Epoch 28/300 -> Train Loss: 0.1377, Train IoU: 0.7736, Val Loss: 0.1347, Val IoU: 0.7801
    Epoch 29/300 -> Train Loss: 0.1328, Train IoU: 0.7783, Val Loss: 0.1294, Val IoU: 0.7875
    Epoch 30/300 -> Train Loss: 0.1270, Train IoU: 0.7890, Val Loss: 0.1234, Val IoU: 0.7983
    Epoch 31/300 -> Train Loss: 0.1206, Train IoU: 0.8029, Val Loss: 0.1183, Val IoU: 0.8046
    Epoch 32/300 -> Train Loss: 0.1152, Train IoU: 0.8119, Val Loss: 0.1133, Val IoU: 0.8133
    Epoch 33/300 -> Train Loss: 0.1140, Train IoU: 0.8121, Val Loss: 0.1197, Val IoU: 0.8024
    Epoch 34/300 -> Train Loss: 0.1163, Train IoU: 0.8029, Val Loss: 0.1121, Val IoU: 0.8148
    Epoch 35/300 -> Train Loss: 0.1142, Train IoU: 0.8080, Val Loss: 0.1064, Val IoU: 0.8241
    Epoch 36/300 -> Train Loss: 0.1093, Train IoU: 0.8149, Val Loss: 0.1099, Val IoU: 0.8186
    Epoch 37/300 -> Train Loss: 0.1094, Train IoU: 0.8156, Val Loss: 0.1059, Val IoU: 0.8250
    Epoch 38/300 -> Train Loss: 0.1067, Train IoU: 0.8193, Val Loss: 0.1050, Val IoU: 0.8263
    Epoch 39/300 -> Train Loss: 0.1044, Train IoU: 0.8249, Val Loss: 0.1063, Val IoU: 0.8253
    Epoch 40/300 -> Train Loss: 0.1036, Train IoU: 0.8243, Val Loss: 0.1040, Val IoU: 0.8247
    Epoch 41/300 -> Train Loss: 0.1013, Train IoU: 0.8283, Val Loss: 0.1037, Val IoU: 0.8293
    Epoch 42/300 -> Train Loss: 0.1006, Train IoU: 0.8290, Val Loss: 0.1010, Val IoU: 0.8294
    Epoch 43/300 -> Train Loss: 0.0984, Train IoU: 0.8328, Val Loss: 0.1002, Val IoU: 0.8356
    Epoch 44/300 -> Train Loss: 0.0978, Train IoU: 0.8342, Val Loss: 0.0980, Val IoU: 0.8382
    Epoch 45/300 -> Train Loss: 0.0959, Train IoU: 0.8396, Val Loss: 0.0967, Val IoU: 0.8409
    Epoch 46/300 -> Train Loss: 0.0946, Train IoU: 0.8410, Val Loss: 0.0970, Val IoU: 0.8441
    Epoch 47/300 -> Train Loss: 0.0947, Train IoU: 0.8421, Val Loss: 0.0951, Val IoU: 0.8453
    Epoch 48/300 -> Train Loss: 0.0932, Train IoU: 0.8452, Val Loss: 0.0943, Val IoU: 0.8456
    Epoch 49/300 -> Train Loss: 0.0923, Train IoU: 0.8460, Val Loss: 0.0946, Val IoU: 0.8480
    Epoch 50/300 -> Train Loss: 0.0925, Train IoU: 0.8458, Val Loss: 0.0947, Val IoU: 0.8485
    Epoch 51/300 -> Train Loss: 0.0926, Train IoU: 0.8463, Val Loss: 0.0931, Val IoU: 0.8482
    Epoch 52/300 -> Train Loss: 0.0912, Train IoU: 0.8486, Val Loss: 0.0927, Val IoU: 0.8480
    Epoch 53/300 -> Train Loss: 0.0905, Train IoU: 0.8491, Val Loss: 0.0928, Val IoU: 0.8500
    Epoch 54/300 -> Train Loss: 0.0906, Train IoU: 0.8487, Val Loss: 0.0932, Val IoU: 0.8501
    Epoch 55/300 -> Train Loss: 0.0908, Train IoU: 0.8485, Val Loss: 0.0922, Val IoU: 0.8500
    Epoch 56/300 -> Train Loss: 0.0899, Train IoU: 0.8506, Val Loss: 0.0916, Val IoU: 0.8480
    Epoch 57/300 -> Train Loss: 0.0894, Train IoU: 0.8506, Val Loss: 0.0913, Val IoU: 0.8510
    Epoch 58/300 -> Train Loss: 0.0890, Train IoU: 0.8509, Val Loss: 0.0917, Val IoU: 0.8523
    Epoch 59/300 -> Train Loss: 0.0892, Train IoU: 0.8508, Val Loss: 0.0908, Val IoU: 0.8524
    Epoch 60/300 -> Train Loss: 0.0883, Train IoU: 0.8532, Val Loss: 0.0897, Val IoU: 0.8515
    Epoch 61/300 -> Train Loss: 0.0879, Train IoU: 0.8533, Val Loss: 0.0895, Val IoU: 0.8531
    Epoch 62/300 -> Train Loss: 0.0870, Train IoU: 0.8540, Val Loss: 0.0891, Val IoU: 0.8564
    Epoch 63/300 -> Train Loss: 0.0875, Train IoU: 0.8543, Val Loss: 0.0890, Val IoU: 0.8549
    Epoch 64/300 -> Train Loss: 0.0862, Train IoU: 0.8563, Val Loss: 0.0876, Val IoU: 0.8555
    Epoch 65/300 -> Train Loss: 0.0862, Train IoU: 0.8560, Val Loss: 0.0876, Val IoU: 0.8566
    Epoch 66/300 -> Train Loss: 0.0860, Train IoU: 0.8559, Val Loss: 0.0869, Val IoU: 0.8569
    Epoch 67/300 -> Train Loss: 0.0861, Train IoU: 0.8562, Val Loss: 0.0867, Val IoU: 0.8565
    Epoch 68/300 -> Train Loss: 0.0843, Train IoU: 0.8588, Val Loss: 0.0859, Val IoU: 0.8587
    Epoch 69/300 -> Train Loss: 0.0852, Train IoU: 0.8572, Val Loss: 0.0860, Val IoU: 0.8603
    Epoch 70/300 -> Train Loss: 0.0839, Train IoU: 0.8600, Val Loss: 0.0859, Val IoU: 0.8587
    Epoch 71/300 -> Train Loss: 0.0848, Train IoU: 0.8580, Val Loss: 0.0856, Val IoU: 0.8604
    Epoch 72/300 -> Train Loss: 0.0833, Train IoU: 0.8602, Val Loss: 0.0847, Val IoU: 0.8597
    Epoch 73/300 -> Train Loss: 0.0838, Train IoU: 0.8596, Val Loss: 0.0850, Val IoU: 0.8607
    Epoch 74/300 -> Train Loss: 0.0825, Train IoU: 0.8613, Val Loss: 0.0845, Val IoU: 0.8606
    Epoch 75/300 -> Train Loss: 0.0830, Train IoU: 0.8606, Val Loss: 0.0844, Val IoU: 0.8605
    Epoch 76/300 -> Train Loss: 0.0822, Train IoU: 0.8615, Val Loss: 0.0843, Val IoU: 0.8616
    Epoch 77/300 -> Train Loss: 0.0827, Train IoU: 0.8612, Val Loss: 0.0842, Val IoU: 0.8621
    Epoch 78/300 -> Train Loss: 0.0824, Train IoU: 0.8617, Val Loss: 0.0839, Val IoU: 0.8613
    Epoch 79/300 -> Train Loss: 0.0818, Train IoU: 0.8625, Val Loss: 0.0842, Val IoU: 0.8591
    Epoch 80/300 -> Train Loss: 0.0822, Train IoU: 0.8611, Val Loss: 0.0840, Val IoU: 0.8624
    Epoch 81/300 -> Train Loss: 0.0812, Train IoU: 0.8630, Val Loss: 0.0836, Val IoU: 0.8633
    Epoch 82/300 -> Train Loss: 0.0818, Train IoU: 0.8630, Val Loss: 0.0838, Val IoU: 0.8625
    Epoch 83/300 -> Train Loss: 0.0819, Train IoU: 0.8620, Val Loss: 0.0841, Val IoU: 0.8616
    Epoch 84/300 -> Train Loss: 0.0814, Train IoU: 0.8631, Val Loss: 0.0836, Val IoU: 0.8599
    Epoch 85/300 -> Train Loss: 0.0818, Train IoU: 0.8617, Val Loss: 0.0838, Val IoU: 0.8609
    Epoch 86/300 -> Train Loss: 0.0806, Train IoU: 0.8634, Val Loss: 0.0833, Val IoU: 0.8639
    Epoch 87/300 -> Train Loss: 0.0808, Train IoU: 0.8640, Val Loss: 0.0834, Val IoU: 0.8624
    Epoch 88/300 -> Train Loss: 0.0804, Train IoU: 0.8649, Val Loss: 0.0831, Val IoU: 0.8611
    Epoch 89/300 -> Train Loss: 0.0809, Train IoU: 0.8627, Val Loss: 0.0836, Val IoU: 0.8628
    Epoch 90/300 -> Train Loss: 0.0798, Train IoU: 0.8650, Val Loss: 0.0829, Val IoU: 0.8620
    Epoch 91/300 -> Train Loss: 0.0803, Train IoU: 0.8646, Val Loss: 0.0822, Val IoU: 0.8635
    Epoch 92/300 -> Train Loss: 0.0801, Train IoU: 0.8641, Val Loss: 0.0838, Val IoU: 0.8642
    Epoch 93/300 -> Train Loss: 0.0802, Train IoU: 0.8651, Val Loss: 0.0835, Val IoU: 0.8605
    Epoch 94/300 -> Train Loss: 0.0802, Train IoU: 0.8649, Val Loss: 0.0826, Val IoU: 0.8618
    Epoch 95/300 -> Train Loss: 0.0803, Train IoU: 0.8631, Val Loss: 0.0862, Val IoU: 0.8628
    Epoch 96/300 -> Train Loss: 0.0820, Train IoU: 0.8620, Val Loss: 0.0821, Val IoU: 0.8647
    Epoch 97/300 -> Train Loss: 0.0802, Train IoU: 0.8652, Val Loss: 0.0825, Val IoU: 0.8620
    Epoch 98/300 -> Train Loss: 0.0804, Train IoU: 0.8638, Val Loss: 0.0827, Val IoU: 0.8633
    Epoch 99/300 -> Train Loss: 0.0797, Train IoU: 0.8649, Val Loss: 0.0822, Val IoU: 0.8644
    Epoch 100/300 -> Train Loss: 0.0797, Train IoU: 0.8658, Val Loss: 0.0822, Val IoU: 0.8625
    Epoch 101/300 -> Train Loss: 0.0796, Train IoU: 0.8651, Val Loss: 0.0823, Val IoU: 0.8632
    Epoch 102/300 -> Train Loss: 0.0786, Train IoU: 0.8666, Val Loss: 0.0815, Val IoU: 0.8651
    Epoch 103/300 -> Train Loss: 0.0789, Train IoU: 0.8665, Val Loss: 0.0820, Val IoU: 0.8638
    Epoch 104/300 -> Train Loss: 0.0787, Train IoU: 0.8668, Val Loss: 0.0816, Val IoU: 0.8626
    Epoch 105/300 -> Train Loss: 0.0789, Train IoU: 0.8662, Val Loss: 0.0813, Val IoU: 0.8646
    Epoch 106/300 -> Train Loss: 0.0784, Train IoU: 0.8669, Val Loss: 0.0809, Val IoU: 0.8669
    Epoch 107/300 -> Train Loss: 0.0779, Train IoU: 0.8684, Val Loss: 0.0820, Val IoU: 0.8626
    Epoch 108/300 -> Train Loss: 0.0786, Train IoU: 0.8673, Val Loss: 0.0811, Val IoU: 0.8638
    Epoch 109/300 -> Train Loss: 0.0794, Train IoU: 0.8649, Val Loss: 0.0830, Val IoU: 0.8639
    Epoch 110/300 -> Train Loss: 0.0775, Train IoU: 0.8691, Val Loss: 0.0808, Val IoU: 0.8645
    Epoch 111/300 -> Train Loss: 0.0796, Train IoU: 0.8656, Val Loss: 0.0822, Val IoU: 0.8639
    Epoch 112/300 -> Train Loss: 0.0774, Train IoU: 0.8687, Val Loss: 0.0817, Val IoU: 0.8646
    Epoch 113/300 -> Train Loss: 0.0790, Train IoU: 0.8667, Val Loss: 0.0826, Val IoU: 0.8633
    Epoch 114/300 -> Train Loss: 0.0777, Train IoU: 0.8684, Val Loss: 0.0803, Val IoU: 0.8658
    Epoch 115/300 -> Train Loss: 0.0786, Train IoU: 0.8672, Val Loss: 0.0813, Val IoU: 0.8649
    Epoch 116/300 -> Train Loss: 0.0775, Train IoU: 0.8685, Val Loss: 0.0807, Val IoU: 0.8649
    Epoch 117/300 -> Train Loss: 0.0785, Train IoU: 0.8669, Val Loss: 0.0819, Val IoU: 0.8635
    Epoch 118/300 -> Train Loss: 0.0780, Train IoU: 0.8679, Val Loss: 0.0799, Val IoU: 0.8668
    Epoch 119/300 -> Train Loss: 0.0773, Train IoU: 0.8690, Val Loss: 0.0803, Val IoU: 0.8657
    Epoch 120/300 -> Train Loss: 0.0778, Train IoU: 0.8683, Val Loss: 0.0806, Val IoU: 0.8649
    Epoch 121/300 -> Train Loss: 0.0774, Train IoU: 0.8686, Val Loss: 0.0797, Val IoU: 0.8660
    Epoch 122/300 -> Train Loss: 0.0773, Train IoU: 0.8685, Val Loss: 0.0798, Val IoU: 0.8678
    Epoch 123/300 -> Train Loss: 0.0760, Train IoU: 0.8711, Val Loss: 0.0791, Val IoU: 0.8680
    Epoch 124/300 -> Train Loss: 0.0768, Train IoU: 0.8699, Val Loss: 0.0805, Val IoU: 0.8641
    Epoch 125/300 -> Train Loss: 0.0770, Train IoU: 0.8692, Val Loss: 0.0788, Val IoU: 0.8678
    Epoch 126/300 -> Train Loss: 0.0762, Train IoU: 0.8704, Val Loss: 0.0786, Val IoU: 0.8699
    Epoch 127/300 -> Train Loss: 0.0756, Train IoU: 0.8718, Val Loss: 0.0789, Val IoU: 0.8681
    Epoch 128/300 -> Train Loss: 0.0763, Train IoU: 0.8707, Val Loss: 0.0795, Val IoU: 0.8656
    Epoch 129/300 -> Train Loss: 0.0763, Train IoU: 0.8701, Val Loss: 0.0783, Val IoU: 0.8701
    Epoch 130/300 -> Train Loss: 0.0753, Train IoU: 0.8723, Val Loss: 0.0785, Val IoU: 0.8708
    Epoch 131/300 -> Train Loss: 0.0759, Train IoU: 0.8715, Val Loss: 0.0787, Val IoU: 0.8699
    Epoch 132/300 -> Train Loss: 0.0759, Train IoU: 0.8715, Val Loss: 0.0790, Val IoU: 0.8661
    Epoch 133/300 -> Train Loss: 0.0768, Train IoU: 0.8701, Val Loss: 0.0796, Val IoU: 0.8647
    Epoch 134/300 -> Train Loss: 0.0764, Train IoU: 0.8699, Val Loss: 0.0781, Val IoU: 0.8700
    Epoch 135/300 -> Train Loss: 0.0749, Train IoU: 0.8728, Val Loss: 0.0783, Val IoU: 0.8714
    Epoch 136/300 -> Train Loss: 0.0743, Train IoU: 0.8743, Val Loss: 0.0783, Val IoU: 0.8683
    Epoch 137/300 -> Train Loss: 0.0764, Train IoU: 0.8711, Val Loss: 0.0794, Val IoU: 0.8647
    Epoch 138/300 -> Train Loss: 0.0765, Train IoU: 0.8697, Val Loss: 0.0779, Val IoU: 0.8695
    Epoch 139/300 -> Train Loss: 0.0746, Train IoU: 0.8732, Val Loss: 0.0780, Val IoU: 0.8710
    Epoch 140/300 -> Train Loss: 0.0743, Train IoU: 0.8740, Val Loss: 0.0776, Val IoU: 0.8709
    Epoch 141/300 -> Train Loss: 0.0748, Train IoU: 0.8737, Val Loss: 0.0784, Val IoU: 0.8672
    Epoch 142/300 -> Train Loss: 0.0760, Train IoU: 0.8708, Val Loss: 0.0781, Val IoU: 0.8679
    Epoch 143/300 -> Train Loss: 0.0746, Train IoU: 0.8730, Val Loss: 0.0772, Val IoU: 0.8726
    Epoch 144/300 -> Train Loss: 0.0742, Train IoU: 0.8742, Val Loss: 0.0766, Val IoU: 0.8739
    Epoch 145/300 -> Train Loss: 0.0738, Train IoU: 0.8753, Val Loss: 0.0779, Val IoU: 0.8681
    Epoch 146/300 -> Train Loss: 0.0766, Train IoU: 0.8710, Val Loss: 0.0801, Val IoU: 0.8621
    Epoch 147/300 -> Train Loss: 0.0762, Train IoU: 0.8700, Val Loss: 0.0767, Val IoU: 0.8738
    Epoch 148/300 -> Train Loss: 0.0730, Train IoU: 0.8757, Val Loss: 0.0769, Val IoU: 0.8740
    Epoch 149/300 -> Train Loss: 0.0732, Train IoU: 0.8767, Val Loss: 0.0773, Val IoU: 0.8692
    Epoch 150/300 -> Train Loss: 0.0752, Train IoU: 0.8730, Val Loss: 0.0770, Val IoU: 0.8698
    Epoch 151/300 -> Train Loss: 0.0737, Train IoU: 0.8743, Val Loss: 0.0759, Val IoU: 0.8751
    Epoch 152/300 -> Train Loss: 0.0722, Train IoU: 0.8776, Val Loss: 0.0758, Val IoU: 0.8750
    Epoch 153/300 -> Train Loss: 0.0726, Train IoU: 0.8769, Val Loss: 0.0761, Val IoU: 0.8735
    Epoch 154/300 -> Train Loss: 0.0734, Train IoU: 0.8763, Val Loss: 0.0785, Val IoU: 0.8646
    Epoch 155/300 -> Train Loss: 0.0762, Train IoU: 0.8709, Val Loss: 0.0759, Val IoU: 0.8728
    Epoch 156/300 -> Train Loss: 0.0726, Train IoU: 0.8761, Val Loss: 0.0763, Val IoU: 0.8758
    Epoch 157/300 -> Train Loss: 0.0719, Train IoU: 0.8786, Val Loss: 0.0754, Val IoU: 0.8733
    Epoch 158/300 -> Train Loss: 0.0726, Train IoU: 0.8775, Val Loss: 0.0760, Val IoU: 0.8706
    Epoch 159/300 -> Train Loss: 0.0731, Train IoU: 0.8757, Val Loss: 0.0747, Val IoU: 0.8744
    Epoch 160/300 -> Train Loss: 0.0716, Train IoU: 0.8785, Val Loss: 0.0744, Val IoU: 0.8758
    Epoch 161/300 -> Train Loss: 0.0715, Train IoU: 0.8787, Val Loss: 0.0743, Val IoU: 0.8770
    Epoch 162/300 -> Train Loss: 0.0702, Train IoU: 0.8811, Val Loss: 0.0735, Val IoU: 0.8773
    Epoch 163/300 -> Train Loss: 0.0724, Train IoU: 0.8776, Val Loss: 0.0753, Val IoU: 0.8743
    Epoch 164/300 -> Train Loss: 0.0720, Train IoU: 0.8780, Val Loss: 0.0747, Val IoU: 0.8737
    Epoch 165/300 -> Train Loss: 0.0735, Train IoU: 0.8765, Val Loss: 0.0820, Val IoU: 0.8583
    Epoch 166/300 -> Train Loss: 0.0757, Train IoU: 0.8712, Val Loss: 0.0759, Val IoU: 0.8712
    Epoch 167/300 -> Train Loss: 0.0725, Train IoU: 0.8770, Val Loss: 0.0762, Val IoU: 0.8763
    Epoch 168/300 -> Train Loss: 0.0706, Train IoU: 0.8805, Val Loss: 0.0739, Val IoU: 0.8772
    Epoch 169/300 -> Train Loss: 0.0715, Train IoU: 0.8796, Val Loss: 0.0754, Val IoU: 0.8709
    Epoch 170/300 -> Train Loss: 0.0722, Train IoU: 0.8778, Val Loss: 0.0742, Val IoU: 0.8744
    Epoch 171/300 -> Train Loss: 0.0711, Train IoU: 0.8788, Val Loss: 0.0743, Val IoU: 0.8789
    Epoch 172/300 -> Train Loss: 0.0696, Train IoU: 0.8821, Val Loss: 0.0733, Val IoU: 0.8782
    Epoch 173/300 -> Train Loss: 0.0707, Train IoU: 0.8814, Val Loss: 0.0791, Val IoU: 0.8621
    Epoch 174/300 -> Train Loss: 0.0745, Train IoU: 0.8736, Val Loss: 0.0744, Val IoU: 0.8749
    Epoch 175/300 -> Train Loss: 0.0708, Train IoU: 0.8790, Val Loss: 0.0747, Val IoU: 0.8790
    Epoch 176/300 -> Train Loss: 0.0702, Train IoU: 0.8817, Val Loss: 0.0745, Val IoU: 0.8751
    Epoch 177/300 -> Train Loss: 0.0717, Train IoU: 0.8796, Val Loss: 0.0744, Val IoU: 0.8727
    Epoch 178/300 -> Train Loss: 0.0713, Train IoU: 0.8784, Val Loss: 0.0724, Val IoU: 0.8807
    Epoch 179/300 -> Train Loss: 0.0690, Train IoU: 0.8835, Val Loss: 0.0723, Val IoU: 0.8814
    Epoch 180/300 -> Train Loss: 0.0682, Train IoU: 0.8845, Val Loss: 0.0721, Val IoU: 0.8805
    Epoch 181/300 -> Train Loss: 0.0693, Train IoU: 0.8834, Val Loss: 0.0745, Val IoU: 0.8714
    Epoch 182/300 -> Train Loss: 0.0725, Train IoU: 0.8770, Val Loss: 0.0725, Val IoU: 0.8759
    Epoch 183/300 -> Train Loss: 0.0704, Train IoU: 0.8800, Val Loss: 0.0743, Val IoU: 0.8798
    Epoch 184/300 -> Train Loss: 0.0685, Train IoU: 0.8838, Val Loss: 0.0712, Val IoU: 0.8812
    Epoch 185/300 -> Train Loss: 0.0680, Train IoU: 0.8857, Val Loss: 0.0719, Val IoU: 0.8770
    Epoch 186/300 -> Train Loss: 0.0679, Train IoU: 0.8841, Val Loss: 0.0693, Val IoU: 0.8847
    Epoch 187/300 -> Train Loss: 0.0659, Train IoU: 0.8880, Val Loss: 0.0698, Val IoU: 0.8851
    Epoch 188/300 -> Train Loss: 0.0655, Train IoU: 0.8891, Val Loss: 0.0690, Val IoU: 0.8839
    Epoch 189/300 -> Train Loss: 0.0662, Train IoU: 0.8877, Val Loss: 0.0713, Val IoU: 0.8767
    Epoch 190/300 -> Train Loss: 0.0675, Train IoU: 0.8852, Val Loss: 0.0684, Val IoU: 0.8831
    Epoch 191/300 -> Train Loss: 0.0672, Train IoU: 0.8857, Val Loss: 0.0693, Val IoU: 0.8857
    Epoch 192/300 -> Train Loss: 0.0653, Train IoU: 0.8889, Val Loss: 0.0686, Val IoU: 0.8867
    Epoch 193/300 -> Train Loss: 0.0647, Train IoU: 0.8906, Val Loss: 0.0686, Val IoU: 0.8827
    Epoch 194/300 -> Train Loss: 0.0675, Train IoU: 0.8857, Val Loss: 0.0703, Val IoU: 0.8786
    Epoch 195/300 -> Train Loss: 0.0672, Train IoU: 0.8855, Val Loss: 0.0702, Val IoU: 0.8856
    Epoch 196/300 -> Train Loss: 0.0655, Train IoU: 0.8885, Val Loss: 0.0694, Val IoU: 0.8868
    Epoch 197/300 -> Train Loss: 0.0663, Train IoU: 0.8898, Val Loss: 0.0750, Val IoU: 0.8689
    Epoch 198/300 -> Train Loss: 0.0701, Train IoU: 0.8797, Val Loss: 0.0708, Val IoU: 0.8835
    Epoch 199/300 -> Train Loss: 0.0666, Train IoU: 0.8863, Val Loss: 0.0691, Val IoU: 0.8866
    Epoch 200/300 -> Train Loss: 0.0674, Train IoU: 0.8879, Val Loss: 0.0696, Val IoU: 0.8800
    Epoch 201/300 -> Train Loss: 0.0664, Train IoU: 0.8851, Val Loss: 0.0705, Val IoU: 0.8850
    Epoch 202/300 -> Train Loss: 0.0646, Train IoU: 0.8912, Val Loss: 0.0664, Val IoU: 0.8866
    Epoch 203/300 -> Train Loss: 0.0642, Train IoU: 0.8913, Val Loss: 0.0654, Val IoU: 0.8908
    Epoch 204/300 -> Train Loss: 0.0623, Train IoU: 0.8934, Val Loss: 0.0646, Val IoU: 0.8927
    Epoch 205/300 -> Train Loss: 0.0619, Train IoU: 0.8958, Val Loss: 0.0655, Val IoU: 0.8872
    Epoch 206/300 -> Train Loss: 0.0644, Train IoU: 0.8902, Val Loss: 0.0668, Val IoU: 0.8887
    Epoch 207/300 -> Train Loss: 0.0630, Train IoU: 0.8925, Val Loss: 0.0652, Val IoU: 0.8920
    Epoch 208/300 -> Train Loss: 0.0629, Train IoU: 0.8935, Val Loss: 0.0645, Val IoU: 0.8901
    Epoch 209/300 -> Train Loss: 0.0622, Train IoU: 0.8939, Val Loss: 0.0641, Val IoU: 0.8922
    Epoch 210/300 -> Train Loss: 0.0615, Train IoU: 0.8955, Val Loss: 0.0639, Val IoU: 0.8943
    Epoch 211/300 -> Train Loss: 0.0619, Train IoU: 0.8945, Val Loss: 0.0627, Val IoU: 0.8949
    Epoch 212/300 -> Train Loss: 0.0609, Train IoU: 0.8970, Val Loss: 0.0657, Val IoU: 0.8866
    Epoch 213/300 -> Train Loss: 0.0622, Train IoU: 0.8937, Val Loss: 0.0612, Val IoU: 0.8975
    Epoch 214/300 -> Train Loss: 0.0605, Train IoU: 0.8971, Val Loss: 0.0650, Val IoU: 0.8929
    Epoch 215/300 -> Train Loss: 0.0603, Train IoU: 0.8977, Val Loss: 0.0619, Val IoU: 0.8951
    Epoch 216/300 -> Train Loss: 0.0615, Train IoU: 0.8956, Val Loss: 0.0640, Val IoU: 0.8897
    Epoch 217/300 -> Train Loss: 0.0612, Train IoU: 0.8953, Val Loss: 0.0606, Val IoU: 0.8973
    Epoch 218/300 -> Train Loss: 0.0605, Train IoU: 0.8966, Val Loss: 0.0643, Val IoU: 0.8938
    Epoch 219/300 -> Train Loss: 0.0596, Train IoU: 0.8989, Val Loss: 0.0614, Val IoU: 0.8956
    Epoch 220/300 -> Train Loss: 0.0601, Train IoU: 0.8979, Val Loss: 0.0604, Val IoU: 0.8967
    Epoch 221/300 -> Train Loss: 0.0590, Train IoU: 0.8992, Val Loss: 0.0619, Val IoU: 0.8965
    Epoch 222/300 -> Train Loss: 0.0581, Train IoU: 0.9011, Val Loss: 0.0601, Val IoU: 0.8994
    Epoch 223/300 -> Train Loss: 0.0579, Train IoU: 0.9018, Val Loss: 0.0610, Val IoU: 0.8946
    Epoch 224/300 -> Train Loss: 0.0583, Train IoU: 0.9001, Val Loss: 0.0590, Val IoU: 0.9010
    Epoch 225/300 -> Train Loss: 0.0565, Train IoU: 0.9032, Val Loss: 0.0583, Val IoU: 0.9025
    Epoch 226/300 -> Train Loss: 0.0567, Train IoU: 0.9039, Val Loss: 0.0596, Val IoU: 0.8968
    Epoch 227/300 -> Train Loss: 0.0576, Train IoU: 0.9009, Val Loss: 0.0562, Val IoU: 0.9051
    Epoch 228/300 -> Train Loss: 0.0559, Train IoU: 0.9045, Val Loss: 0.0573, Val IoU: 0.9042
    Epoch 229/300 -> Train Loss: 0.0552, Train IoU: 0.9062, Val Loss: 0.0565, Val IoU: 0.9024
    Epoch 230/300 -> Train Loss: 0.0560, Train IoU: 0.9043, Val Loss: 0.0555, Val IoU: 0.9052
    Epoch 231/300 -> Train Loss: 0.0545, Train IoU: 0.9061, Val Loss: 0.0554, Val IoU: 0.9070
    Epoch 232/300 -> Train Loss: 0.0545, Train IoU: 0.9068, Val Loss: 0.0538, Val IoU: 0.9090
    Epoch 233/300 -> Train Loss: 0.0538, Train IoU: 0.9082, Val Loss: 0.0562, Val IoU: 0.9036
    Epoch 234/300 -> Train Loss: 0.0544, Train IoU: 0.9068, Val Loss: 0.0527, Val IoU: 0.9103
    Epoch 235/300 -> Train Loss: 0.0529, Train IoU: 0.9092, Val Loss: 0.0535, Val IoU: 0.9101
    Epoch 236/300 -> Train Loss: 0.0525, Train IoU: 0.9102, Val Loss: 0.0523, Val IoU: 0.9116
    Epoch 237/300 -> Train Loss: 0.0519, Train IoU: 0.9114, Val Loss: 0.0528, Val IoU: 0.9090
    Epoch 238/300 -> Train Loss: 0.0530, Train IoU: 0.9091, Val Loss: 0.0518, Val IoU: 0.9114
    Epoch 239/300 -> Train Loss: 0.0519, Train IoU: 0.9103, Val Loss: 0.0536, Val IoU: 0.9104
    Epoch 240/300 -> Train Loss: 0.0514, Train IoU: 0.9119, Val Loss: 0.0504, Val IoU: 0.9142
    Epoch 241/300 -> Train Loss: 0.0520, Train IoU: 0.9113, Val Loss: 0.0522, Val IoU: 0.9098
    Epoch 242/300 -> Train Loss: 0.0519, Train IoU: 0.9105, Val Loss: 0.0501, Val IoU: 0.9151
    Epoch 243/300 -> Train Loss: 0.0501, Train IoU: 0.9137, Val Loss: 0.0495, Val IoU: 0.9159
    Epoch 244/300 -> Train Loss: 0.0499, Train IoU: 0.9147, Val Loss: 0.0501, Val IoU: 0.9131
    Epoch 245/300 -> Train Loss: 0.0504, Train IoU: 0.9130, Val Loss: 0.0482, Val IoU: 0.9179
    Epoch 246/300 -> Train Loss: 0.0492, Train IoU: 0.9150, Val Loss: 0.0488, Val IoU: 0.9174
    Epoch 247/300 -> Train Loss: 0.0488, Train IoU: 0.9166, Val Loss: 0.0475, Val IoU: 0.9180
    Epoch 248/300 -> Train Loss: 0.0490, Train IoU: 0.9155, Val Loss: 0.0478, Val IoU: 0.9179
    Epoch 249/300 -> Train Loss: 0.0483, Train IoU: 0.9165, Val Loss: 0.0466, Val IoU: 0.9205
    Epoch 250/300 -> Train Loss: 0.0477, Train IoU: 0.9175, Val Loss: 0.0467, Val IoU: 0.9206
    Epoch 251/300 -> Train Loss: 0.0475, Train IoU: 0.9181, Val Loss: 0.0461, Val IoU: 0.9209
    Epoch 252/300 -> Train Loss: 0.0477, Train IoU: 0.9180, Val Loss: 0.0483, Val IoU: 0.9158
    Epoch 253/300 -> Train Loss: 0.0499, Train IoU: 0.9142, Val Loss: 0.0456, Val IoU: 0.9221
    Epoch 254/300 -> Train Loss: 0.0481, Train IoU: 0.9167, Val Loss: 0.0487, Val IoU: 0.9182
    Epoch 255/300 -> Train Loss: 0.0487, Train IoU: 0.9165, Val Loss: 0.0453, Val IoU: 0.9220
    Epoch 256/300 -> Train Loss: 0.0479, Train IoU: 0.9180, Val Loss: 0.0473, Val IoU: 0.9177
    Epoch 257/300 -> Train Loss: 0.0486, Train IoU: 0.9157, Val Loss: 0.0467, Val IoU: 0.9209
    Epoch 258/300 -> Train Loss: 0.0482, Train IoU: 0.9166, Val Loss: 0.0450, Val IoU: 0.9230
    Epoch 259/300 -> Train Loss: 0.0475, Train IoU: 0.9188, Val Loss: 0.0477, Val IoU: 0.9175
    Epoch 260/300 -> Train Loss: 0.0478, Train IoU: 0.9172, Val Loss: 0.0457, Val IoU: 0.9223
    Epoch 261/300 -> Train Loss: 0.0463, Train IoU: 0.9203, Val Loss: 0.0444, Val IoU: 0.9233
    Epoch 262/300 -> Train Loss: 0.0460, Train IoU: 0.9203, Val Loss: 0.0437, Val IoU: 0.9250
    Epoch 263/300 -> Train Loss: 0.0455, Train IoU: 0.9212, Val Loss: 0.0431, Val IoU: 0.9261
    Epoch 264/300 -> Train Loss: 0.0449, Train IoU: 0.9225, Val Loss: 0.0426, Val IoU: 0.9267
    Epoch 265/300 -> Train Loss: 0.0447, Train IoU: 0.9226, Val Loss: 0.0423, Val IoU: 0.9273
    Epoch 266/300 -> Train Loss: 0.0442, Train IoU: 0.9235, Val Loss: 0.0422, Val IoU: 0.9276
    Epoch 267/300 -> Train Loss: 0.0440, Train IoU: 0.9239, Val Loss: 0.0421, Val IoU: 0.9273
    Epoch 268/300 -> Train Loss: 0.0443, Train IoU: 0.9234, Val Loss: 0.0412, Val IoU: 0.9292
    Epoch 269/300 -> Train Loss: 0.0434, Train IoU: 0.9245, Val Loss: 0.0416, Val IoU: 0.9287
    Epoch 270/300 -> Train Loss: 0.0435, Train IoU: 0.9245, Val Loss: 0.0409, Val IoU: 0.9295
    Epoch 271/300 -> Train Loss: 0.0434, Train IoU: 0.9251, Val Loss: 0.0417, Val IoU: 0.9277
    Epoch 272/300 -> Train Loss: 0.0443, Train IoU: 0.9232, Val Loss: 0.0409, Val IoU: 0.9295
    Epoch 273/300 -> Train Loss: 0.0430, Train IoU: 0.9252, Val Loss: 0.0412, Val IoU: 0.9294
    Epoch 274/300 -> Train Loss: 0.0433, Train IoU: 0.9251, Val Loss: 0.0401, Val IoU: 0.9306
    Epoch 275/300 -> Train Loss: 0.0427, Train IoU: 0.9264, Val Loss: 0.0408, Val IoU: 0.9289
    Epoch 276/300 -> Train Loss: 0.0432, Train IoU: 0.9249, Val Loss: 0.0397, Val IoU: 0.9317
    Epoch 277/300 -> Train Loss: 0.0424, Train IoU: 0.9264, Val Loss: 0.0391, Val IoU: 0.9325
    Epoch 278/300 -> Train Loss: 0.0422, Train IoU: 0.9272, Val Loss: 0.0397, Val IoU: 0.9311
    Epoch 279/300 -> Train Loss: 0.0419, Train IoU: 0.9272, Val Loss: 0.0387, Val IoU: 0.9331
    Epoch 280/300 -> Train Loss: 0.0413, Train IoU: 0.9282, Val Loss: 0.0384, Val IoU: 0.9334
    Epoch 281/300 -> Train Loss: 0.0412, Train IoU: 0.9287, Val Loss: 0.0386, Val IoU: 0.9326
    Epoch 282/300 -> Train Loss: 0.0413, Train IoU: 0.9281, Val Loss: 0.0378, Val IoU: 0.9346
    Epoch 283/300 -> Train Loss: 0.0404, Train IoU: 0.9296, Val Loss: 0.0379, Val IoU: 0.9347
    Epoch 284/300 -> Train Loss: 0.0405, Train IoU: 0.9297, Val Loss: 0.0371, Val IoU: 0.9356
    Epoch 285/300 -> Train Loss: 0.0402, Train IoU: 0.9301, Val Loss: 0.0376, Val IoU: 0.9349
    Epoch 286/300 -> Train Loss: 0.0403, Train IoU: 0.9299, Val Loss: 0.0370, Val IoU: 0.9358
    Epoch 287/300 -> Train Loss: 0.0399, Train IoU: 0.9306, Val Loss: 0.0376, Val IoU: 0.9353
    Epoch 288/300 -> Train Loss: 0.0398, Train IoU: 0.9309, Val Loss: 0.0361, Val IoU: 0.9372
    Epoch 289/300 -> Train Loss: 0.0394, Train IoU: 0.9316, Val Loss: 0.0376, Val IoU: 0.9343
    Epoch 290/300 -> Train Loss: 0.0402, Train IoU: 0.9301, Val Loss: 0.0367, Val IoU: 0.9361
    Epoch 291/300 -> Train Loss: 0.0394, Train IoU: 0.9313, Val Loss: 0.0362, Val IoU: 0.9373
    Epoch 292/300 -> Train Loss: 0.0395, Train IoU: 0.9313, Val Loss: 0.0356, Val IoU: 0.9381
    Epoch 293/300 -> Train Loss: 0.0390, Train IoU: 0.9326, Val Loss: 0.0367, Val IoU: 0.9358
    Epoch 294/300 -> Train Loss: 0.0389, Train IoU: 0.9323, Val Loss: 0.0350, Val IoU: 0.9391
    Epoch 295/300 -> Train Loss: 0.0383, Train IoU: 0.9332, Val Loss: 0.0350, Val IoU: 0.9394
    Epoch 296/300 -> Train Loss: 0.0382, Train IoU: 0.9336, Val Loss: 0.0345, Val IoU: 0.9402
    Epoch 297/300 -> Train Loss: 0.0381, Train IoU: 0.9340, Val Loss: 0.0359, Val IoU: 0.9371
    Epoch 298/300 -> Train Loss: 0.0388, Train IoU: 0.9323, Val Loss: 0.0342, Val IoU: 0.9410
    Epoch 299/300 -> Train Loss: 0.0378, Train IoU: 0.9341, Val Loss: 0.0353, Val IoU: 0.9394
    Epoch 300/300 -> Train Loss: 0.0383, Train IoU: 0.9334, Val Loss: 0.0340, Val IoU: 0.9411
    

![download 1.png](/assets/images/post_data/UNet_Implementation_image/download%201.png)

You can see that the loss is lower, and the IoU is higher compared to the case without a validation set.

- IoU(Intersection over Union)
    
    A metric used in object detection and segmentation tasks, IoU represents the overlap between the predicted bounding box and the ground truth. The value ranges between 0 and 1.
    

**Prediction results for test image (image #8)**

![8.png](/assets/images/post_data/UNet_Implementation_image/8.png)

![prediction_8 (1).png](/assets/images/post_data/UNet_Implementation_image/prediction_8_(1).png)

When comparing the test image predictions between cases with and without a validation set, it appears that the prediction with a validation set produces a cleaner output with better separation of the membrane.