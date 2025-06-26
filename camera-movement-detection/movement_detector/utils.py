import cv2
import numpy as np
import base64
from typing import Any

def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def calculate_frame_difference_score(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    try:
        diff = cv2.absdiff(prev_frame, curr_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        change_percentage = (changed_pixels / total_pixels) * 100
        mean_diff = np.mean(diff)
        return float(change_percentage + mean_diff * 0.1)
    except:
        return 0.0

def calculate_optical_flow_score(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    try:
        corners = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if corners is not None and len(corners) > 10:
            next_corners, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, corners, None)
            good_old = corners[status == 1]
            good_new = next_corners[status == 1]
            if len(good_old) > 5:
                displacements = np.linalg.norm(good_new - good_old, axis=1)
                return float(np.median(displacements) * 3)
        return 0.0
    except:
        return 0.0

def calculate_edge_motion_score(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    try:
        edges_prev = cv2.Canny(prev_frame, 50, 150)
        edges_curr = cv2.Canny(curr_frame, 50, 150)
        edge_diff = cv2.absdiff(edges_prev, edges_curr)
        edge_motion = np.mean(edge_diff)
        return float(edge_motion * 0.5)
    except:
        return 0.0 