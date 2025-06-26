# scoring.py
import numpy as np
from typing import Any, Dict

def calculate_movement_score(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    if H is not None:
        try:
            translation_x = H[0, 2]
            translation_y = H[1, 2]
            translation_magnitude = np.sqrt(translation_x**2 + translation_y**2)
            rotation_angle = np.arctan2(H[1, 0], H[0, 0])
            rotation_magnitude = np.abs(rotation_angle)
            scale_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
            scale_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
            scale_change = np.abs(1.0 - (scale_x + scale_y) / 2)
            perspective_dist = np.abs(H[2, 0]) + np.abs(H[2, 1])
            if len(src_pts) > 0 and len(dst_pts) > 0:
                displacements = np.linalg.norm(dst_pts.reshape(-1, 2) - src_pts.reshape(-1, 2), axis=1)
                displacement_variance = np.var(displacements) if len(displacements) > 1 else 0
            else:
                displacement_variance = 0
            movement_score = (
                translation_magnitude * 1.5 +
                rotation_magnitude * 30 +
                scale_change * 50 +
                perspective_dist * 20 +
                displacement_variance * 0.1
            )
            return float(movement_score)
        except:
            return 0.0
    else:
        try:
            if len(src_pts) == 0 or len(dst_pts) == 0:
                return 0.0
            displacements = np.linalg.norm(dst_pts.reshape(-1, 2) - src_pts.reshape(-1, 2), axis=1)
            median_displacement = np.median(displacements)
            displacement_variance = np.var(displacements)
            return float(median_displacement + displacement_variance * 0.1)
        except:
            return 0.0

def classify_movement_type(transformation_matrix: np.ndarray) -> Dict[str, float]:
    if transformation_matrix is None:
        return {
            'translation_x': 0.0,
            'translation_y': 0.0,
            'rotation': 0.0,
            'scaling': 1.0,
            'perspective': 0.0
        }
    translation_x = transformation_matrix[0, 2]
    translation_y = transformation_matrix[1, 2]
    rotation = np.degrees(np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0]))
    scaling_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[1, 0]**2)
    scaling_y = np.sqrt(transformation_matrix[0, 1]**2 + transformation_matrix[1, 1]**2)
    scaling = (scaling_x + scaling_y) / 2
    perspective = np.abs(transformation_matrix[2, 0]) + np.abs(transformation_matrix[2, 1])
    return {
        'translation_x': float(translation_x),
        'translation_y': float(translation_y),
        'rotation': float(rotation),
        'scaling': float(scaling),
        'perspective': float(perspective)
    } 