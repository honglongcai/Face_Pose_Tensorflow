# Face Pose Estimation

## Getting Started
 This project represents a real-time multi-person face-pose estimation method. 
 It only applies three small convolutional layers which makes it a super fast 
 real time face pose estimation method.

## Prerequest:

 Tensorflow v1
 
 OpenCV

## Dataset
 Download data used in training and evaluation phase in this link
 [dataset](https://drive.google.com/file/d/1CT2EiXcrta3452hqISWTXpSoeyukqiTR/view?usp=sharing).
 
 This is LFW dataset and I labeled the face pose of each image using landmarks and SVD.
 You can find code used to label images in landmarks_pose directory. 
 
 90% images are useded in training and 10% useding in evaluation. 
 
 Please follow LFW license when using this dataset.

## Training steps:
  
 1. mkdir data 
 
 2. unzip download dataset to data directory
 
 3. mkdir model
 
 4. python train.py
 
## TODO:
 Add SavedModel module which makes stored model easily used.

## Example Video:

[![Watch the video](https://img.youtube.com/vi/QG5eheTpjNc/0.jpg)](https://www.youtube.com/embed/QG5eheTpjNc)

## License:

 MIT LICENSE
