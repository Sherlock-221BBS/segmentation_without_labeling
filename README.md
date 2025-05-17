# Segmentation Without Labeling

This repository presents contain code to do perform semantic segmentation without the labourious and tedious task of labelling images. All you have to do is have a reference image containing the object that you want to segement and its corresponding mask and you are set to do segmentation for that particular object in all the images. You only have to label once. You may ask, **how could this be achieved?**.

The answer to it is using SAM and CLIP or some other VLM. The methodology is explained below - 

## Methodology 
Our approach consists of four main components: 
1. A reference image and its segmentation mask (Item A)
2. A textual description of the target class (Item B)
3. A set of unlabeled images (Item C); and 
4. A test set with ground truth segmentation masks (Item D). 

The goal is to use Items
A and B to generate pseudo-labels for Item C, fine-tune a
segmentation model on these pseudo-labels, and evaluate the
model on Item D. The most important part is creating good quality pseduo labels. 

Let's understand how could this be achieved step by step - 

### Psudo Label Creation

To generate pseudo-labels for the unlabeled images in Item
C, we use a combination of SAM and CLIP. For each image
in Item C, we perform the following steps: 
1) We stitch the reference image (Item A) and the target
image side by side. 
2) We identify foreground and background
points in the reference image using its mask. 
3) We use CLIP
to identify regions in the target image that are semantically
similar to the textual description of your class. For example for segmenting cat - ”a photo of a cat.” 
4) We use
these points as prompts for SAM to generate a segmentation
mask for the target image.
5) We select the mask with the
highest confidence score as the pseudo-label for the target
image.

This is the most important step. Once we have the labels genereated, we can **finetune any SOTA segmentation model** on these images and achieve satisfactory performance. 

You could visualize the performance of this methdology here - 

[1](./segmentation_result_cat_043.png)

[2](./segmentation_result_cat_274.png)

