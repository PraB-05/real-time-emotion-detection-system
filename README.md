# Real-Time Emotion Detection Using CNN

This project is about detecting human facial emotions in real time using a webcam. I built a custom Convolutional Neural Network (CNN) using TensorFlow/Keras and integrated it with OpenCV to detect faces and predict their emotions from live video.

The model can classify five different emotions: **Anger, Fear, Happy, Neutral, and Sad.**

## About the Project

The main goal of this project was to build an emotion classification model and use it in a real-time application.

I trained the CNN on a dataset of 12,023 facial images and then used the trained model to make predictions through a webcam.

**Some details about the project:**
- Total images: 12,023
- Training images: 9,619
- Testing images: 2,404
- Image size: 128 × 128 × 3
- Number of emotion classes: 5
- Test accuracy: 93.26%

## Dataset

The dataset contains images belonging to five emotion categories.

| Emotion | Test Images |
|---|---:|
| Anger | 250 |
| Fear | 491 |
| Happy | 533 |
| Neutral | 562 |
| Sad | 568 |
| **Total** | **2,404** |

Before training, I resized the images to 128 × 128 pixels and normalized their pixel values between 0 and 1.

I also used image augmentation to introduce some variation into the training data:

- Shear range: 0.2
- Horizontal flip
- Zoom range: 0.15
- Rescaling: 1/255

## CNN Model

I built a CNN with three convolutional layers to learn different facial features from the images.

The model architecture consists of:

1. Conv2D layer with 32 filters and ReLU activation.
2. MaxPooling layer (2 × 2).
3. Conv2D layer with 64 filters and ReLU activation.
4. MaxPooling layer (2 × 2).
5. Conv2D layer with 128 filters and ReLU activation.
6. MaxPooling layer (2 × 2).
7. Flatten layer.
8. Dense layer with 128 neurons.
9. Output layer with 5 neurons and Softmax activation.

### Training Configuration

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- EarlyStopping: Monitored validation loss
- Patience: 7 epochs

I used EarlyStopping to prevent unnecessary training once the validation loss stopped improving.

## Results

After training and evaluating the model on the test dataset, I achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **93.26%** |
| Test Loss | **0.20** |

## Real-Time Emotion Detection

Once the model was trained, I integrated it with OpenCV to make predictions using a webcam.

Here's how the application works:

1. OpenCV captures frames from the webcam.
2. Haar Cascade is used to detect faces in each frame.
3. The detected face is cropped from the original frame.
4. The cropped image is resized to 128 × 128 pixels.
5. The image is converted from BGR to RGB.
6. Pixel values are normalized between 0 and 1.
7. The processed image is passed to the trained CNN.
8. The predicted emotion is displayed on the webcam feed.

Press `q` to close the webcam window.

## Project Structure

```text
Real-Time-Emotion-Detection/
│
├── Emotion_detection.ipynb
├── real_time_emotion_prediction.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── README.md
└── Emotion_detection_model.keras
```

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/Real-Time-Emotion-Detection.git
```

### Step 2: Navigate to the project folder

```bash
cd Real-Time-Emotion-Detection
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Download the trained model

The trained model is over 25 MB, so I haven't uploaded it directly to GitHub.

Download it using the link below and place it in the project folder.

[Download Emotion_detection_model.keras](https://drive.google.com/drive/folders/1iyTBxznOyveBpXa_RqrSncahp0N_UkdD?usp=sharing)

### Step 5: Run the application

```bash
python real_time_emotion_prediction.py
```

Make sure your webcam is connected before running the script.

## Technologies Used

- Python
- TensorFlow and Keras
- Convolutional Neural Networks (CNN)
- OpenCV
- NumPy
- Matplotlib
- Pandas

## What I Learned

While working on this project, I got hands-on experience with:

- Preparing and preprocessing image datasets.
- Building and training a CNN from scratch.
- Applying image augmentation during training.
- Evaluating a classification model on test data.
- Loading and using a saved Keras model.
- Working with OpenCV for face detection.
- Integrating a deep learning model into a real-time webcam application.

---

**Thanks for checking out my project!**
