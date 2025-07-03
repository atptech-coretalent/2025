# 📷 Camera Movement Detector

This project was developed as part of the **ATP Core Talent AI Coder Challenge 2025**.  
It is a Streamlit-based web application that analyzes uploaded videos and detects **significant camera movements** such as panning, tilting, or shaking.  
It does **not** focus on object motion — instead, it detects changes in the global scene indicating that the camera itself moved.

---

## 📷 Screenshot

![Camera Detector Screenshot](https://github.com/user-attachments/assets/90a2831b-2218-42a6-8172-f3ee63661157)

## 🚀 Live App

🔗 [Click here to try the app online](https://your-deployment-url.com)  
(Replace this with your Streamlit or Hugging Face link after deployment)

---

## 🎯 Features

- 📼 Upload a video (MP4, AVI, MOV)
- 🎞️ Extracts and analyzes frames
- 🎯 Detects significant camera movement (not object motion)
- 💡 Clean, modern UI with ATP branding

---

## 🧠 Approach

- **Frame Extraction:** Extracts all frames from the video using OpenCV.
- **Feature Detection & Matching:** ORB feature detector is used to compare consecutive frames.
- **Homography Transformation:** Calculates transformation matrix to estimate global scene shift.
- **Movement Decision:** If the transformation deviates above a threshold, it is classified as camera movement.
- **Bonus Logic:** Tries to filter out object motion by comparing matched keypoints and transformation consistency.

---

## 🛠️ Tech Stack

- **Python 3.13**
- **OpenCV**
- **Streamlit** (for UI)
- **Datasets** from Hugging Face (for CameraBench demo data)
- **NumPy** for numerical processing

---

## 📦 Installation

To run locally:

```bash
git clone https://github.com/yourusername/camera-detector.git
cd camera-detector
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure

```bash
camera-detector/
├── app.py                 # Streamlit web UI
├── movement_detector.py   # Core detection logic
├── requirements.txt       # Dependencies
├── README.md              # This file
```

### 📸 Example Output

Input: Sample video with camera shake

#### Output:

```bash

📌 Detected camera movement at frame indices: [6, 7, 8, 10, 15, 22, 23, 24]
```
## 💬 Acknowledgements

* Hugging Face CameraBench Dataset
* ATP Core Talent Team for organizing the challenge

## 🧑‍💻 Developed by

* Saadet Elizaveta Babal
* Candidate – ATP Core Talent AI Coder Challenge 2025

