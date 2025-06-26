from .features import extract_features, match_features
from .scoring import calculate_movement_score, classify_movement_type
from .utils import calculate_frame_difference_score, calculate_optical_flow_score, calculate_edge_motion_score
import numpy as np
import cv2
from typing import List, Dict, Any

class CameraMovementDetector:
    def __init__(self, threshold_feature: float = 5.0, threshold_homography: float = 15.0, min_match_count: int = 8):
        self.threshold_feature = threshold_feature
        self.threshold_homography = threshold_homography
        self.min_match_count = min_match_count

    def detect(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        movement_indices = []
        movement_scores = []
        transformation_matrices = []
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        prev_frame = None
        prev_kp = None
        prev_des = None
        for idx, frame in enumerate(frames):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            kp, des = orb.detectAndCompute(gray, None)
            if prev_frame is not None and prev_des is not None and des is not None:
                movement_score = 0
                H = None
                if len(prev_des) > 0 and len(des) > 0:
                    matches = bf.match(prev_des, des)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = []
                    if len(matches) > 0:
                        distance_threshold = min(50, matches[0].distance * 2.5)
                        good_matches = [m for m in matches if m.distance < distance_threshold]
                    if len(good_matches) >= self.min_match_count:
                        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=1000)
                        if H is not None:
                            movement_score = calculate_movement_score(H, src_pts, dst_pts)
                            if movement_score > self.threshold_homography:
                                movement_indices.append(idx)
                        else:
                            movement_score = calculate_movement_score(None, src_pts, dst_pts)
                            if movement_score > self.threshold_homography * 0.5:
                                movement_indices.append(idx)
                    if movement_score == 0 or len(good_matches) < self.min_match_count:
                        scores = []
                        diff_score = calculate_frame_difference_score(prev_frame, gray)
                        scores.append(diff_score)
                        flow_score = calculate_optical_flow_score(prev_frame, gray)
                        scores.append(flow_score)
                        edge_score = calculate_edge_motion_score(prev_frame, gray)
                        scores.append(edge_score)
                        movement_score = max(scores) if scores else 0
                        if movement_score > self.threshold_feature:
                            movement_indices.append(idx)
                    movement_scores.append(movement_score)
                    transformation_matrices.append(H)
                else:
                    movement_scores.append(0)
                    transformation_matrices.append(None)
            else:
                if idx > 0:
                    movement_scores.append(0)
                    transformation_matrices.append(None)
            prev_frame = gray
            prev_kp = kp
            prev_des = des
        return {
            'movement_indices': movement_indices,
            'movement_scores': movement_scores,
            'transformation_matrices': transformation_matrices
        } 