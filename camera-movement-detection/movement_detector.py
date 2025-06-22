import cv2 
import numpy as np
from typing import List

class CameraMovementDetector:
     def __init__(self, threshold=0.5, min_match_count=10, debug=False):
        self.threshold = threshold
        self.min_match_count = min_match_count
        self.debug = debug
        self.detector = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.last_movement_info = {}


     def detect(self, frame1: np.ndarray, frame2: np.ndarray) -> tuple[bool, float, dict[str, any]]:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < self.min_match_count or len(kp2) < self.min_match_count:
            if self.debug:
                print("Yetersiz anahtar nokta")
            return False, 0.0, {"error": "Not enough keypoints"}

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < self.min_match_count:
            if self.debug:
                print("Yetersiz iyi eşleşme")
            return False, 0.0, {"error": "Not enough good matches"}

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            if self.debug:
                print("Homografi hesaplanamadı")
            return False, 0.0, {"error": "Homography could not be computed"}

        dx, dy = H[0, 2], H[1, 2]
        translation = np.sqrt(dx**2 + dy**2)
        det = H[0, 0]*H[1, 1] - H[0, 1]*H[1, 0]
        identity_diff = np.sum(np.abs(H - np.eye(3)))

        inliers = mask.ravel().sum() if mask is not None else 0
        inlier_ratio = inliers / len(good_matches) if good_matches else 0

        movement_score = (0.5 * translation + 0.3 * abs(1-det) + 0.2 * identity_diff) * inlier_ratio
        is_movement = movement_score > self.threshold

        self.last_movement_info = {
            "translation": translation,
            "determinant": det,
            "identity_diff": identity_diff,
            "movement_score": movement_score,
            "num_matches": len(good_matches),
            "inlier_ratio": inlier_ratio
        }

        return is_movement, movement_score, self.last_movement_info


if __name__ == "__main__":
    # Test 
    frame1 = cv2.imread("frame1.png")
    frame2 = cv2.imread("frame2.jpg")

    if frame1 is None or frame2 is None:
        print("Lütfen 'frame1.jpg' ve 'frame2.jpg' dosyalarını aynı klasöre koyun.")
        exit(1)

    detector = CameraMovementDetector(threshold=0.5, min_match_count=10, debug=True)
    is_movement, movement_score, details = detector.detect(frame1, frame2)

    print("Kamera hareketi tespit edildi mi?:", is_movement)
    print("Hareket skoru:", movement_score)
    print("Detaylar:", details)

#Verilen fonksiyon
def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    """
    Detect frames where significant camera movement occurs.
    Args:
        frames: List of image frames (as numpy arrays).
        threshold: Sensitivity threshold for detecting movement.
    Returns:
        List of indices where significant movement is detected.
    """
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



