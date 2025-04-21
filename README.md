# ASL Recognition with Convolutional Neural Network

## Overview
This project provides an advanced American Sign Language (ASL) recognition system that translates hand gestures into text in real-time. Using a combination of pre-trained Convolutional Neural Networks (CNN) and Random Forest Classifier, the system achieves over 95% accuracy in recognizing ASL symbols.

## Technical Features
- Leverages Google's Mediapipe BlazePalm detector for precise hand landmark recognition.</br>
- Interprets ASL gestures through 2D hand position tracking.</br>
- Trained using scikit-learn on a comprehensive dataset of ASL signs.</br>
- Processes webcam input for real-time communication assistance.</br>

## Requirements
- Webcam access. </br>
- Python with required libraries (Mediapipe, scikit-learn, OpenCV).</br>

## Setup and Usage Instructions
1. Data Collection: Run `collect_data.py` to capture training images from your webcam.</br>
2. Training the Model: Execute `create_dataset.py` followed by `train_classifier.py` to prepare and train the classifier.</br>
3. Real-time Recognition: Launch `inference_classifier.py` to begin detecting and translating ASL symbols in real-time.</br>
