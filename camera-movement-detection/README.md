# Camera Movement Detection

This application detects significant camera movement in a sequence of images or video. It distinguishes between object movement within the scene and actual camera movement (tilt, pan, large translation).

## Overview

The application uses computer vision techniques to analyze consecutive frames and detect when significant camera movement occurs. It employs feature matching and homography estimation to identify camera movements while attempting to distinguish them from object movements within the scene.

## Features

- **Image Sequence Analysis**: Upload and analyze a sequence of images
- **Video Analysis**: Upload and analyze video files
- **Movement Detection**: Identifies frames with significant camera movement
- **Movement Classification**: Classifies the type of camera movement (translation, rotation, scaling)
- **Interactive Visualization**: Displays movement scores and detected frames
- **Adjustable Parameters**: Fine-tune detection sensitivity

## Technical Approach

### Movement Detection Logic

The system uses a two-tier approach to detect camera movements:

1. **Feature-based Detection**:
   - Extracts ORB (Oriented FAST and Rotated BRIEF) features from consecutive frames
   - Matches these features using FLANN (Fast Library for Approximate Nearest Neighbors)
   - Applies ratio test to filter good matches

2. **Homography Estimation**:
   - Calculates homography matrix between matched features using RANSAC
   - Decomposes the matrix to extract movement parameters:
     - Translation (x, y)
     - Rotation angle
     - Scaling factor
     - Perspective distortion

3. **Fallback Method**:
   - If not enough feature matches are found, falls back to simple frame differencing
   - Calculates mean absolute difference between consecutive frames

### Movement Classification

The system classifies camera movements into the following types:
- **Translation**: Horizontal and vertical camera movement
- **Rotation**: Camera rotation around optical axis
- **Scaling**: Change in zoom level or camera distance
- **Perspective**: Change in camera viewing angle

## Installation

### Prerequisites

- Python 3.8 or higher

### Setup

1. Clone the repository:
```
git clone <repository-url>
cd camera-movement-detection
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Running Locally

Start the Streamlit app:
```
streamlit run app.py
```

The application will be available at http://localhost:8501

### Using the App

1. **Upload Content**:
   - Choose the "Image Sequence" tab to upload multiple image files
   - Choose the "Video" tab to upload a video file

2. **Adjust Parameters** (optional):
   - Feature Threshold: Sensitivity for pixel-based detection
   - Homography Threshold: Sensitivity for feature-based detection
   - Minimum Feature Matches: Required matches for homography estimation
   - Sample Rate: For video processing, analyze every Nth frame

3. **Run Detection**:
   - Click "Detect Movement in Images" or "Detect Movement in Video"
   - View the results in the movement score graph and detected frames

## Example Results

The application provides:
- A graph showing movement scores across all frames
- A list of frames where significant movement was detected
- Detailed analysis of each detected movement
- Visual display of the frames with detected movement

## Limitations and Assumptions

- The system works best with high-contrast scenes with distinct features
- Very fast movements may be missed if using higher sample rates with video
- Extremely low-light conditions may reduce detection accuracy
- The system assumes the camera is generally stationary with occasional movements

## Future Improvements

- Implement deep learning-based movement detection
- Add support for real-time webcam analysis
- Improve distinction between object and camera movement
- Add batch processing for multiple videos

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Modular Structure (2024+)

The codebase is now organized as a Python package for better modularity and maintainability:

- `movement_detector/detector.py`: Main detection logic and CameraMovementDetector class
- `movement_detector/features.py`: Feature extraction and matching
- `movement_detector/scoring.py`: Movement scoring and classification
- `movement_detector/utils.py`: Utility functions

**Note:** `movement_detector.py` is deprecated. Please use the new package structure.

# Example Usage

```python
from movement_detector.detector import CameraMovementDetector
from movement_detector.scoring import classify_movement_type

detector = CameraMovementDetector(threshold_feature=5.0, threshold_homography=15.0, min_match_count=8)
result = detector.detect(frames)
```
