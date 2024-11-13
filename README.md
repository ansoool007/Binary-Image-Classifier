# Binary Image Classifier using CNN

## Overview
This project focuses on developing a **Binary Image Classifier** using Convolutional Neural Networks (CNNs). The classifier is trained to distinguish between images of cats and dogs using a dataset from Kaggle. The model achieved an impressive validation accuracy of **99.7%** on a larger dataset. Techniques like **Data Augmentation** and **Dropout Layers** were employed to prevent overfitting, particularly when dealing with smaller datasets.

## Introduction
Binary image classification is a fundamental task in computer vision where the goal is to classify images into one of two categories. This project employs Convolutional Neural Networks (CNNs) to build a robust classifier, achieving high accuracy by leveraging modern techniques and tools like **Keras**, **Matplotlib**, and **NumPy**.

## Dataset
- **Source**: [Cats and Dogs image dataset from Kaggle](https://www.kaggle.com/).
- **Type**: Binary classification (Cats vs. Dogs)
- **Dataset Size**: A mix of smaller and larger subsets to evaluate performance.
- **Image Size**: Resized for CNN input compatibility.

## Data Preprocessing
To optimize model performance, the following preprocessing steps were carried out:
- **Data Augmentation**: Applied transformations like rotation, flipping, and zooming to increase dataset diversity.
- **Normalization**: Pixel values were scaled to a range of [0, 1].
- **Dropout Layers**: Added during model training to reduce the risk of overfitting on smaller datasets.

## Model Architecture
The CNN architecture includes:
- **Convolutional Layers**: To extract features from input images.
- **Pooling Layers**: Max Pooling to reduce the dimensionality of feature maps.
- **Dropout Layers**: Introduced to prevent overfitting during training.
- **Fully Connected Layers**: For making predictions based on the extracted features.
- **Activation Functions**: ReLU for hidden layers and Sigmoid for binary output classification.

### Key Features:
- **Data Augmentation**: Used to enhance the dataset, creating variability and reducing overfitting.
- **Dropout**: Implemented for robust training, particularly on smaller datasets.
- **Loss Function**: `binary_crossentropy` for binary classification tasks.
- **Optimizer**: Adam Optimizer for efficient gradient descent.

## Results
The model delivered excellent results:
- **Validation Accuracy**: 99.7% on a larger dataset.
- **Overfitting Control**: Successfully mitigated with the help of data augmentation and dropout strategies.
