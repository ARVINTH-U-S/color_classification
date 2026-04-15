# 🎨 Color Classification Models

This repository contains implementations of multiple machine learning
and deep learning approaches for **color classification**, including:

-   CNN-based model\
-   EfficientNet-based model\
-   MLP-based model

Each model has separate scripts for **training** and **testing**.

------------------------------------------------------------------------

## 📂 Project Structure

    color_classification/
    │
    ├── cnn_color_classification/
    │   ├── cnn_training.py
    │   ├── cnn_testing.py
    │
    ├── efficientnet_color_classification/
    │   ├── efficientnet_training.py
    │   ├── efficientnet_testing.py
    │
    ├── mlp_color_classification/
    │   ├── mlp_training.py
    │   ├── mlp_testing.py
    │
    └── README.md

------------------------------------------------------------------------

## 🚀 Features

-   Multiple model architectures for comparison\
-   Modular structure (separate training & testing scripts)\
-   Easy to extend for new models\
-   Supports experimentation with different approaches

------------------------------------------------------------------------

## 🧠 Models Included

### 1. CNN Model

-   Custom Convolutional Neural Network\
-   Suitable for basic image-based color classification

### 2. EfficientNet Model

-   Transfer learning using EfficientNet\
-   Better performance with optimized architecture

### 3. MLP Model

-   Fully connected neural network\
-   Works on flattened image features

------------------------------------------------------------------------

## 🏋️ Training

### CNN

``` bash
python cnn_color_classification/cnn_training.py
```

### EfficientNet

``` bash
python efficientnet_color_classification/efficientnet_training.py
```

### MLP

``` bash
python mlp_color_classification/mlp_training.py
```

------------------------------------------------------------------------

## 🧪 Testing

### CNN

``` bash
python cnn_color_classification/cnn_testing.py
```

### EfficientNet

``` bash
python efficientnet_color_classification/efficientnet_testing.py
```

### MLP

``` bash
python mlp_color_classification/mlp_testing.py
```

------------------------------------------------------------------------

## Dataset Structure

    dataset/
    ├── red/
    ├── blue/
    ├── green/
    ├── ...
