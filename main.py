import time
import cv2
from src.face_detector import FaceDetector
from src.emotion_detector import EmotionDetector
from src.config import RESULTS_DIR


def open_camera():
    # Try default index, followed by DirectShow on Windows, then index 1
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)
    return camera


def main():
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    camera = open_camera()

    if not camera.isOpened():
        raise RuntimeError('Could not open webcam. Please verify your camera is connected and accessible.')

    previous_time = time.time()
    print('Press Q or ESC to quit. Press S to save a frame.')

    while True:
        success, frame = camera.read()
        if not success:
            break

        faces = face_detector.detect(frame)
        for (x, y, w, h) in faces:
            y1, y2 = max(0, y), min(frame.shape[0], y + h)
            x1, x2 = max(0, x), min(frame.shape[1], x + w)
            if y2 <= y1 or x2 <= x1:
                continue

            face = frame[y1:y2, x1:x2]
            try:
                emotion, confidence, _ = emotion_detector.predict(face)
            except Exception:
                emotion, confidence = 'Unknown', 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{emotion} {confidence * 100:.1f}%' if confidence > 0 else f'{emotion}'
            cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()
        fps = 1 / max(now - previous_time, 1e-6)
        previous_time = now

        cv2.putText(frame, f'Faces: {len(faces)}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not emotion_detector.is_loaded:
            cv2.putText(frame, 'Face Tracking Active (Run train.py for emotions)', (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow('Facial Emotion Detection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('s'), ord('S')):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = RESULTS_DIR / f'capture_{int(time.time())}.jpg'
            cv2.imwrite(str(filename), frame)
            print('Saved:', filename)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

