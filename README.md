# Real-Time Facial Emotion Detection

A lightweight computer-vision project that detects faces from a webcam and predicts visible facial-expression classes using a TensorFlow/Keras CNN trained on FER2013.

## Features
- Real-time webcam emotion detection
- OpenCV face detection
- Seven emotion classes
- Confidence score and FPS
- Multiple face detection
- Image prediction
- Model training and evaluation
- Confusion matrix and training graph

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download FER2013 and place it at `data/fer2013.csv`. The CSV must contain `emotion` and `pixels` columns.

Dataset: https://www.kaggle.com/datasets/msambare/fer2013

## Train
```bash
python train.py
```
Use the full dataset with `python train.py --limit 0`.

## Run webcam
```bash
python main.py
```
Press Q/ESC to quit and S to save a frame.

## Predict an image
```bash
python predict_image.py sample.jpg
```

## Evaluate
```bash
python evaluate.py
```

## Pipeline
Webcam/Image -> Face Detection -> Face Crop -> Grayscale/Resize -> CNN -> Emotion + Confidence

## Classes
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Note
The classifier predicts visible facial-expression patterns; it does not establish a person's actual internal emotional state. Lighting, pose, occlusion, camera quality and dataset bias can affect predictions.

## Author
SANJAY P T
