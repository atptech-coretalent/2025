import os

# Streamlit config dosyasını kullanıcının home dizininde oluştur
config_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
os.makedirs(config_dir, exist_ok=True)

config_path = os.path.join(config_dir, "config.toml")
with open(config_path, "w") as f:
    f.write("""
[server]
headless = true
port = $PORT
enableCORS = false
""")

import streamlit as st
import tempfile
from movement_detector import extract_frames, detect_camera_movement

st.set_page_config(page_title="Camera Movement Detector", layout="centered")

# CSS + HTML (Profesyonel Tasarım)
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background-color: #f4f4f4;
        padding-top: 30px;
    }

    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
    }

    .logo-container img {
        width: 60px;
        height: auto;
        opacity: 0.9;
    }

    .card {
        background-color: white;
        border-radius: 12px;
        padding: 30px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    .title {
        text-align: center;
        font-size: 26px;
        color: #333;
        margin-bottom: 10px;
        font-weight: 600;
    }

    .description {
        text-align: center;
        font-size: 16px;
        color: #666;
        margin-bottom: 25px;
    }

    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: bold;
    }

    .stFileUploader label {
        font-weight: 500;
        color: #444;
    }

    /* 🔧 Info ve spinner kutularının metin rengini siyah yap */
    .stAlert > div {
        color: black !important;
    }

    .stSpinner > div {
        color: black !important;
    }

    </style>
""", unsafe_allow_html=True)

# 🔹 Logo
st.markdown("""
<div class="logo-container">
    <img src="https://i.hizliresim.com/khnekdt.png" alt="ATP Logo">
</div>
""", unsafe_allow_html=True)

# 🔹 İçerik Kartı
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="title">📷 Camera Movement Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload a video to detect <b>significant camera movement</b> (not object motion).</div>', unsafe_allow_html=True)

# 🔸 Dosya yükleyici
uploaded_video = st.file_uploader("🎞️ Upload your video", type=["mp4", "avi", "mov"])

# 🔸 Analiz işlemi
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    st.info("Extracting frames...")
    frames = extract_frames(temp_video_path)

    st.info("Analyzing for camera movement...")
    with st.spinner("Running detection..."):
        indices = detect_camera_movement(frames)

    st.success("✅ Detection complete.")
    st.markdown(f"<div class='description'>📌 Movement detected at frames:<br><code>{indices}</code></div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
