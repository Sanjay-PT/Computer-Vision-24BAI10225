import argparse
import cv2
from src.face_detector import FaceDetector
from src.emotion_detector import EmotionDetector


def main():
    parser = argparse.ArgumentParser(description='Detect face and predict emotion on a static image.')
    parser.add_argument('image', help='Path to input image file')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f'Could not open image: {args.image}')

    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    faces = face_detector.detect(image)

    if len(faces) == 0:
        print('No face detected in the image.')
        return

    print(f'Detected {len(faces)} face(s):')
    for n, (x, y, w, h) in enumerate(faces, 1):
        y1, y2 = max(0, y), min(image.shape[0], y + h)
        x1, x2 = max(0, x), min(image.shape[1], x + w)
        if y2 <= y1 or x2 <= x1:
            continue

        face = image[y1:y2, x1:x2]
        try:
            emotion, conf, _ = emotion_detector.predict(face)
        except Exception:
            emotion, conf = 'Unknown', 0.0

        if conf > 0:
            print(f'Face {n}: {emotion} ({conf * 100:.2f}%)')
            label = f'{emotion} {conf * 100:.1f}%'
        else:
            print(f'Face {n}: {emotion}')
            label = f'{emotion}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

