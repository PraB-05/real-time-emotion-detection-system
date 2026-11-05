Real-Time Emotion Detection
A computer vision project that classifies facial emotions live from a webcam using a custom CNN.
It predicts 5 emotions — Anger, Fear, Happy, Neutral, and Sad — trained on 12,023 facial images (9,619 train / 2,404 test).
Dataset
Total images: 12,023
Train: 9,619 images
Test: 2,404 images
Classes: Anger (250 test), Fear (491 test), Happy (533 test), Neutral (562 test), Sad (568 test)
Image size: 128x128x3
Augmentation: shear 0.2, horizontal flip, zoom 0.15, rescale 1/255
Model
Built with TensorFlow/Keras.
Input: 128x128x3
Conv2D 32 filters, 3x3, ReLU, MaxPool 2x2
Conv2D 64 filters, 3x3, ReLU, MaxPool 2x2
Conv2D 128 filters, 3x3, ReLU, MaxPool 2x2
Flatten
Dense 128, ReLU
Dense 5, softmax
Optimizer: Adam
Loss: sparse categorical crossentropy
EarlyStopping on val_loss, patience 7
Results
Test accuracy: 93.26%
Test loss: 0.20
Real-time pipeline
OpenCV + Haar Cascade detects the face from the webcam, crops and resizes it to 128x128, converts BGR to RGB, rescales to 0-1, and overlays the predicted emotion live on the video feed. Press q to quit.
The trained model is saved as Emotion_detection_model.keras.
Project structure
Emotion_detection.ipynb
real_time_emotion_prediction.py
haarcascade_frontalface_default.xml
README.md
requirements.txt
How to run
pip install -r requirements.txt
python real_time_emotion_prediction.py
Requirements
tensorflow
opencv-contrib-python
numpy
matplotlib
pandas
Model download
The trained .keras model file is over 25 MB, so it is not uploaded to GitHub. Download it from the link below and place it in the project folder before running the real-time script:
[Download Emotion_detection_model.keras — add your Google Drive link here]
Tech stack
Python, TensorFlow/Keras, CNN, OpenCV, NumPy, Matplotlib
