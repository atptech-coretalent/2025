import cv2
import numpy as np
from typing import Any

def extract_features(frame: np.ndarray) -> Any:
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des

def match_features(des1, des2) -> Any:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    return [] 