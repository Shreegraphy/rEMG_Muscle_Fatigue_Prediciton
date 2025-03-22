# Muscle Fatigue Prediction using RNN and EMG Data

## Overview
This project implements a Long Short-Term Memory (LSTM) based Recurrent Neural Network (RNN) model to predict muscle fatigue using electromyography (EMG) data. The model leverages time-series EMG signals to classify fatigue and non-fatigue states, aiming to improve injury prevention, athletic performance, and industrial safety.

## Features
- **Data Preprocessing**: Standardization, handling missing values, and reshaping data for RNN input.
- **Deep Learning Model**: LSTM-based sequential model for time-series classification.
- **Performance Metrics**: Accuracy, Precision, Recall, Confusion Matrix, and F1-score.
- **Visualization**: Training curves and confusion matrix for model performance analysis.
- **Real-time Prediction**: User input-based fatigue prediction using a trained model.

## Dataset
The dataset comprises EMG signal recordings with corresponding fatigue labels. The data is preprocessed to normalize values and reshape them for sequential modeling.

**Dataset Link**: [Google Drive Dataset](https://drive.google.com/drive/folders/1Hg9t7KSs_RR9A6RlPkDssbq4IuQ7l3bA?usp=sharing)

## Model Architecture
The model consists of:
- LSTM layers with 128 and 64 units
- Dropout layers to prevent overfitting
- Dense layers with ReLU activation
- Output layer with a sigmoid activation for binary classification

## Installation
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

## Usage
### Clone the Repository:
```bash
git clone https://github.com/yourusername/muscle-fatigue-rnn.git
cd muscle-fatigue-rnn
```

### Load Data
Ensure your EMG data files are in the designated folder.

### Train Model
Run the script to preprocess data and train the LSTM model.
```bash
python train_model.py
```

### Evaluate Performance
View accuracy, loss plots, and confusion matrix.

### Predict Fatigue
Use the interactive script to input EMG data and receive fatigue predictions.
```bash
python predict.py
```

## Results
The LSTM model achieves a high classification accuracy (~99.65%), demonstrating its reliability in detecting muscle fatigue. The confusion matrix and classification reports validate its robustness.

## Future Improvements
- Multi-channel EMG data integration
- Exploring attention mechanisms for enhanced performance
- Application of the model in real-time wearable monitoring systems

## References
This project is inspired by recent research in muscle fatigue detection, including the IEEE paper _"Muscle Fatigue Detection Using Wearable EMG Data and Long Short-Term Memory Networks."_
