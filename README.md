# Diagnosis-of-bone-fractures-with-F1-0.91
Diagnosing Human Bones Fractures and Non-Fractures with images F1-Score=0.91 - With DenseNet121 Model
# Diagnosis of Bone Fractures using DenseNet121 (F1-Score: 0.91)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg?logo=linkedin)](https://www.linkedin.com/in/your-linkedin-profile/)  <!-- جایگزین با لینک واقعی کنید -->

This project demonstrates a deep learning model for detecting bone fractures in X-ray images using transfer learning with DenseNet121. Achieving an F1-score of 0.91 on the non-fractured class and strong overall performance, it's a practical example of applying AI in medical imaging.

## Project Overview

Bone fractures are common injuries, and early detection via X-rays can save lives. This Jupyter notebook builds a binary classifier (fractured vs. non-fractured) using convolutional neural networks (CNNs). 

- **For Beginners**: We'll walk through the basics of image classification, data augmentation, and model training step-by-step. No prior deep learning experience needed—just follow along!
- **For Experts**: Dive into transfer learning, fine-tuning, handling class imbalance, and evaluation metrics like F1-score for imbalanced datasets.
- **Key Achievement**: Model accuracy of 89.06% on test set, with F1-scores of 0.84 (fractured) and 0.92 (non-fractured), addressing real-world medical challenges.

The notebook is self-contained and runs on platforms like Kaggle or Google Colab.

## Features

- **Dataset**: Human bone X-ray images from Kaggle [](https://www.kaggle.com/datasets/your-dataset-link). Split into train (294 images), validation (62), and test (64) sets. Classes: Fractured and Non-Fractured.
- **Model**: Pre-trained DenseNet121 from Keras, with custom layers for binary classification.
- **Techniques Used**:
  - Data augmentation to handle small dataset.
  - Class weighting for imbalance (more non-fractured images).
  - Two-stage training: Transfer learning + Fine-tuning.
- **Results**: High recall on fractured class (0.90), making it useful for medical screening.
- **Visualizations**: Training plots, confusion matrix, and sample predictions.

## Prerequisites

To run this notebook, you'll need:
- Python 3.8+
- Libraries: TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Seaborn.
- Dataset: Download from Kaggle and place in `/kaggle/input/` (or adjust paths for local run).

Install dependencies via:
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
