### Author: Kenneth Zhang

## Semantic Segmentation for Aerial Drone Dataset (Research Group 2)

### Semantic image segmentation is a branch of computer vision and its goal is to label each pixel of an image with a corresponding class of what is being represented. The output in semantic  segmentation is a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. It is a pixel level image classification.

## Examples of the applications of this task are:
### Autonomous vehicles, where semantic segmentation provides information about free space on the roads, as well as to detect lane markings and traffic signs. Biomedical image diagnosis,  helping radiologists improving analysis performed, greatly reducing the time required to run diagnostic tests. Geo sensing, to recognize the type of land cover (e.g., areas of urban,  agriculture, water, etc.) for each pixel on a satellite image, land cover classification can be regarded as a multi-class semantic segmentation task.

## U-Net Convolutional architecture (FCN-Variant):
### U-Net is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.  The architecture of a U-Net contains two paths: the first one is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a   traditional stack of convolutional and max pooling layers; the second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using  transposed convolutions. In the original paper, the U-Net is described as follows:

<img src = "https://miro.medium.com/max/3000/1*OkUrpDD6I0FpugA_bbYBJQ.png" width = 600px/>
