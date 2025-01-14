# Plant Disease Classification using Deep Learning

## Overview
This project implements a deep learning model for classifying plant diseases across 38 different categories. Using transfer learning with the Xception architecture, the model can effectively identify various plant diseases from images, providing a valuable tool for agricultural disease management.

## Performance Metrics
- Overall Accuracy: 98%
- Weighted Average Precision: 98%
- Perfect AUC scores for multiple disease categories
- Individual disease performance:
  - Apple Scab: 98% F1-score
  - Tomato Yellow Leaf Curl Virus: 100% F1-score
  - Grape Black Rot: 98% F1-score
  - Most categories achieving >95% F1-score

## Technical Features
- **Model Architecture**: Transfer learning with Xception
- **Training Optimizations**:
  - Dynamic learning rate adjustment
  - Model rollback capabilities
  - Progressive layer unfreezing
  - Stratified sampling for balanced training
- **Data Processing**:
  - Support for multiple image formats (color, grayscale, segmented)
  - Advanced augmentation techniques
  - Balanced dataset handling

## Model Architecture
- Base Model: Xception (pre-trained on ImageNet)
- Additional Layers:
  - Global Average Pooling
  - Dense Layer (1024 units) with ReLU activation
  - Dropout Layer (0.5)
  - Dense Layer (512 units) with ReLU activation
  - Dropout Layer (0.3)
  - Output Layer (38 units) with Softmax activation

## Dataset Structure
The dataset is organized into three main directories:
- `/color`: Original RGB images
- `/grayscale`: Grayscale versions of the images
- `/segmented`: Segmented versions of the images

## Requirements
```
tensorflow>=2.0.0
numpy
pandas
opencv-python
scikit-learn
matplotlib
seaborn
tqdm
```

## Training Process
1. **Initial Phase**
   - Base model layers frozen
   - Training of top layers
   - Learning rate: 1e-3

2. **Fine-tuning Phase**
   - Progressive unfreezing of layers
   - Reduced learning rate: 1e-5
   - Dynamic learning rate adjustment
   - Model rollback on performance degradation

3. **Optimization Features**
   - Early stopping mechanism
   - Learning rate reduction on plateau
   - Model checkpointing
   - Stratified data sampling

## Performance Details
- Training Accuracy: 99.68%
- Validation Accuracy: 98.25%
- Test Accuracy: 98%
- AUC Score: Perfect (1.0) for multiple categories

## Training Results Visualization
```
Final Training Metrics:
- Accuracy: 99.68%
- AUC: 1.0000
- Loss: 0.0112

Final Validation Metrics:
- Accuracy: 98.25%
- AUC: 0.9978
- Loss: 0.0661
```
