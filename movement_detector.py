import cv2
import numpy as np

def extract_frames(video_path, resize=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames

def detect_camera_movement(frames, threshold=15):
    orb = cv2.ORB_create()
    movement_indices = []

    for i in range(len(frames) - 1):
        kp1, des1 = orb.detectAndCompute(frames[i], None)
        kp2, des2 = orb.detectAndCompute(frames[i+1], None)

        if des1 is None or des2 is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Homografi matrisi hesaplanıyor
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # Homografi’den dönüşüm şiddetini tahmin ediyoruz
            dx = H[0, 2]
            dy = H[1, 2]
            movement_magnitude = np.sqrt(dx**2 + dy**2)

            if movement_magnitude > threshold:
                movement_indices.append(i+1)

    return movement_indices
