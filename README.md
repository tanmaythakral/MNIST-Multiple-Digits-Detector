![image](https://github.com/tanmaythakral/MNIST-Multiple-Digits-Detector/assets/46583379/7d12806e-b26e-41e2-bf55-09250a3d34f4)

# MNIST ResNet with MSER Image Processing

This repository contains a PyTorch implementation of a ResNet model trained on the MNIST dataset with additional image processing using MSER (Maximally Stable Extremal Regions).

## Features

- **ResNet Model:** The repository includes a ResNet18 architecture implemented in PyTorch for image classification on the MNIST dataset.
  
- **MSER Image Processing:** The code utilizes MSER processing to extract regions of interest from images, which are then fed into the ResNet model for classification.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- torchvision
- PIL (Pillow)

## Image Processing Pipeline
- MSER processing is performed on input images to extract regions of interest (ROIs).
- Each ROI is cropped, padded, and resized to match the input size of the ResNet model.

## Results
The trained ResNet model achieves an impressive accuracy of 96% on the test set, showcasing its effectiveness in handwritten digit recognition.

## Acknowledgments
- The ResNet architecture is inspired by the work of Kaiming He et al. (arXiv:1512.03385).
- MSER processing is implemented using OpenCV.
