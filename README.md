# Drowsiness Detection System

A robust drowsiness detection system using deep learning to analyze eye states and yawning for driver safety monitoring.


## Overview

This project implements a drowsiness detection system using deep learning to analyze facial features. The system classifies images into four categories ("Open", "Closed", "yawn", and "no_yawn") and provides a comprehensive drowsiness assessment to help prevent accidents caused by driver fatigue.

## Features

- **EfficientNetB0-based Model:** Leverages a pre-trained EfficientNetB0 architecture fine-tuned for drowsiness detection
- **Eye Region Analysis:** Uses advanced cropping and quality assessment algorithms for precise eye state classification
- **Multi-Class Classification:** Predicts four distinct classes related to eye state and yawning
- **Combined Prediction Strategy:** Enhances accuracy by fusing predictions from full face and cropped eye regions
- **Data Augmentation:** Implements robust augmentation techniques to improve model generalization
- **Comprehensive Evaluation:** Provides detailed performance metrics including accuracy, precision, recall, and F1-score

## Requirements

- Python 3.10+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- Scikit-learn
- NumPy
- Pathlib

### Installation

```bash
# Clone the repository
git clone https://github.com/username/drowsiness-detection.git
cd drowsiness-detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

The model requires a dataset organized in the following structure:

```
dataset_new/
├── train/
│   ├── Open/
│   ├── Closed/
│   ├── yawn/
│   └── no_yawn/
└── test/
    ├── Open/
    ├── Closed/
    ├── yawn/
    └── no_yawn/
```

Each subdirectory should contain corresponding labeled facial images.

[Dataset](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new) taken from kaggle. 

## Usage

### Training the Model

```bash
python train_model.py
```

This will train the model using the dataset in the `dataset_new` directory and save the trained model as `best_model.h5`.

### Testing on a Single Image

```bash
python validate_model.py --image_path path/to/your/image.jpg
```


## Project Structure

```
drowsiness-detection/
├── train_model.py           # Script for training the model
├── validate_model.py        # Script for testing on individual images
├── eye_crop_best.py              # Functions for eye region extraction and analysis
├── best_model.h5            # Trained model weights
├── requirements.txt         # Dependencies
├── dataset_new/             # Training and testing datasets
├── testing_images/          # Sample images for testing                
├── accuracy_loss.png        # Training/validation curves
└── confusion_matrix.png     # Confusion matrix visualization
```

## Results

The model achieved 100% accuracy on the test set with the following performance metrics:

| Class    | Precision% | Recall% |
|----------|-----------|--------|
| Open     | 100     | 100   |
| Closed   | 100      | 100   |
| yawn     | 99      | 100   |
| no_yawn  | 100     | 99  | 


## Future Work

- **Real-time Optimization:** Enhance processing speed for smoother real-time performance
- **Alert System Integration:** Develop configurable alert mechanisms (sound, vibration) with sensitivity controls
- **Environmental Robustness:** Improve performance under varying lighting conditions and with users wearing glasses
- **Mobile Deployment:** Create optimized versions for deployment on mobile and embedded devices
- **Multi-modal Analysis:** Incorporate additional biometric signals for more robust drowsiness detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- EfficientNet implementation based on [TensorFlow's EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- Eye detection methodology inspired by research paper: "Driver drowsiness detection and smart alerting using deep learning and IoT" (2023)
- To my project mate: Telna Chacko and Arshida
