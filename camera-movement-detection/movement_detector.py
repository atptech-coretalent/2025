import cv2
import numpy as np
from typing import List

def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    movement_indices = []
    prev_gray = None
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff)
            if score > threshold:
                movement_indices.append(idx)
        prev_gray = gray
    return movement_indices

def detect_camera_movement_orb(frames: List[np.ndarray], movement_threshold: float = 30.0) -> List[int]:
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    movement_indices = []
    prev_kp, prev_des = None, None

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if (
            prev_des is not None and des is not None and
            prev_kp is not None and kp is not None and
            len(prev_kp) > 0 and len(kp) > 0
        ):
            matches = bf.match(prev_des, des)
            if len(matches) > 0:
               
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                if len(src_pts) >= 4 and len(dst_pts) >= 4:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        dx = H[0, 2]
                        dy = H[1, 2]
                        da = np.arctan2(H[1, 0], H[0, 0])
                        movement = np.sqrt(dx**2 + dy**2) + np.abs(da) * 50 
                        if movement > movement_threshold:
                            movement_indices.append(idx)
        prev_kp, prev_des = kp, des
    return movement_indices
