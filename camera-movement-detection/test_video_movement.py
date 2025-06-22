import cv2
from movement_detector import detect_camera_movement_orb


cap = cv2.VideoCapture("testVideo.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()


movement_indices = detect_camera_movement_orb(frames, movement_threshold=30.0)
print("Kamera hareketi tespit edilen frame indexleri:", movement_indices) 