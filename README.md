# Image-super-resolution using Enhanced Deep Residual Networks for Single Image Super-Resolution(EDSR) and Wide Activation for Efficient and Accurate Image Super-Resolution(WDSR)

Image Super Resolution using EDSR and WDSR research papers

**You can find Deployed Working model [here](http://ec2-54-236-46-15.compute-1.amazonaws.com:8080/)**

**Find detailed blog of my project [here](https://sumittagadiya.medium.com/image-super-resolution-using-edsr-and-wdsr-f4de0b00e039)**

**Note : If page is not loading then please paste above ipynb github link [here](https://nbviewer.jupyter.org/). Now you can view ipynb notebook successfully !!**

# Demo of Working Model
![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/working_demo.gif?raw=true)

# EDSR 
### Architecture of EDSR :

![alt text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/Architecture/EDSR.png?raw=true)
* EDSR paper is [here](https://arxiv.org/pdf/1707.02921.pdf)

### Architecture of Residual Block
![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/Architecture/res_block.png?raw=true)

* Proposed networks. We remove the batch normalization layers from our network as Nah et al presented in their image deblurring work. Since batch normalization layers normalize the features, they get rid of range flexibility from networks by normalizing the features, it is better to remove them.

* Furthermore, GPU memory usage is also sufficiently reduced since the batch normalization layers consume the same amount of memory as the preceding convolutional
layers. this baseline model without batch normalization layer saves approximately 40% of memory usage during training, compared to SRResNet. Consequently, we can
build up a larger model that has better performance than conventional ResNet structure under limited computational resources.

# WDSR
### Architecture of WDSR :
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/Architecture/WDSR_architecture.png?raw=true)
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/Architecture/wdsr.png?raw=true)
* WDSR paper is [here](https://arxiv.org/pdf/1808.08718.pdf)

# Dataset
* [DIV2K](https://www.tensorflow.org/datasets/catalog/div2k) dataset is a newly proposed high-quality
(2K resolution) image dataset for image restoration tasks. The DIV2K dataset consists of 800 training images, 100
validation images, and 100 test images. As the test dataset ground truth is not released, we report and compare the performances on the validation dataset. We also compare the performance on some of the standard benchmark datasets named Set5 and  Set14.

# Results of EDSR 
### 1. Bicubic_x4 (Bicubic downgrading with scaling factor = 4)
#### Div2k validation set results
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/bicubic_results/div2k_bicubic_x4/download4.png?raw=true)
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/bicubic_results/div2k_bicubic_x4/download.png?raw=true)
* More result you can find [here](https://github.com/sumittagadiya/Image-super-resolution/tree/main/predicted_images/EDSR/bicubic_results/div2k_bicubic_x4)

#### Set 5 Bicubic_x4
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/bicubic_results/set_5/set_5_bicubic_x4/download.png?raw=true)
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/bicubic_results/set_5/set_5_bicubic_x4/download2.png?raw=true)
* More result you can find [here](https://github.com/sumittagadiya/Image-super-resolution/tree/main/predicted_images/EDSR/bicubic_results/set_5/set_5_bicubic_x4)

#### Set14 Bicubic_x4
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/bicubic_results/set_5/set_14_bicubic_x4/9.png?raw=true)
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/bicubic_results/set_5/set_14_bicubic_x4/4.png?raw=true)
* More result you can find [here](https://github.com/sumittagadiya/Image-super-resolution/tree/main/predicted_images/EDSR/bicubic_results/set_5/set_14_bicubic_x4)

### 2. Unknown_x4(Unknown downgrading with scaling factor = 4)
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/unknown_x4/download1.png?raw=true)
* ![alt_text](https://github.com/sumittagadiya/Image-super-resolution/blob/main/predicted_images/EDSR/unknown_x4/download4.png?raw=true)
* More result you can find [here](https://github.com/sumittagadiya/Image-super-resolution/tree/main/predicted_images/EDSR/unknown_x4)

