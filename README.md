# README
# U-Net Model for Semantic Segmentation

This repository contains the code for a U-Net-based model designed for single-class semantic segmentation, particularly optimized for UAV (Unmanned Aerial Vehicle) datasets. The model is implemented with and without data augmentation, allowing for flexibility in testing the impact of augmentation on segmentation performance.

## Overview

The U-Net model is a popular convolutional neural network (CNN) architecture tailored for image segmentation tasks. It employs an encoder-decoder structure with skip connections, preserving spatial information across layers and facilitating precise localization of the segmented regions. The model in this repository is optimized and adapted to UAV dataset specifications with the potential for modifications to better suit specific use cases.  

**Note:** The UAV dataset utilized for this model is private and restricted from public access. 

## Table of Contents

- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training and Evaluation](#training-and-evaluation)
- [Visualization](#visualization)
- [License and Acknowledgments](#license-and-acknowledgments)

## Model Architecture

The U-Net model follows a standard U-Net design with an encoder path, a bottleneck layer, and a decoder path. Skip connections between layers in the encoder and decoder paths help retain spatial context:

1. **Encoder Path**: Consists of several convolutional layers with ReLU activation and max pooling layers, which progressively reduce spatial resolution while capturing features.
2. **Bottleneck**: Acts as a bridge between the encoder and decoder paths, capturing high-level features.
3. **Decoder Path**: Uses upsampling layers to gradually restore spatial resolution, with skip connections added from corresponding encoder layers to retain context.
4. **Output Layer**: A 1x1 convolution with a sigmoid activation outputs the final segmentation mask.

```python
# Model Instantiation
model = unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Data Augmentation

Data augmentation is essential in scenarios where datasets may be limited. The U-Net model is trained with and without data augmentation using the following techniques:

- **Rotation**: Random rotation within a specified range.
- **Width and Height Shifts**: Random translation along horizontal and vertical axes.
- **Zoom**: Scaling the image by a random factor.
- **Horizontal Flip**: Flipping the image horizontally.
- **Rescaling**: Normalizing pixel values.

The `ImageDataGenerator` class in TensorFlow is used to apply these augmentations. The results can be compared to evaluate the model's performance improvement due to augmentation.

## Training and Evaluation

The training process is configured with various callbacks to monitor the model's performance and prevent overfitting:

- **ModelCheckpoint**: Saves the best model based on validation loss.
- **EarlyStopping**: Stops training when validation loss does not improve.
- **Visualization Callback**: Generates and saves sample predictions for each epoch, aiding in visual inspection of the model's learning progress.

**Evaluation Metrics**:
- **Precision**
- **F1 Score**
- **Mean IoU (Intersection over Union)**

Training and validation loss/accuracy are saved and visualized to help analyze model performance across epochs. Confusion matrices and inference times are also computed to offer insights into the model's accuracy and speed.

## Visualization

Sample predictions are saved per epoch, allowing a visual comparison between ground truth masks and the model's predicted masks. These visualizations can help in assessing the model's performance qualitatively.

## License and Acknowledgments

This code is licensed under the MIT License. The U-Net model architecture is based on the original paper: **[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)** by Ronneberger et al.

Attributions for any used libraries, including TensorFlow, Keras, and OpenCV, belong to their respective authors.

---

**Disclaimer**: Due to modifications made to the original U-Net architecture to suit the UAV dataset, results may vary. The dataset used in this project is proprietary and not publicly accessible.
```