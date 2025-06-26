import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import matplotlib.pyplot as plt
from PIL import Image
from movement_detector.detector import CameraMovementDetector
from movement_detector.scoring import classify_movement_type
from typing import List, Dict, Any
import base64
from datetime import datetime
from movement_detector.ui_helpers import (
    plot_movement_scores, display_movement_details, display_all_frames, create_simple_report, image_to_base64,
    get_custom_css, get_title_html, get_about_card_html, get_info_box_html, get_footer_html
)

# Set page configuration with custom theme
st.set_page_config(
    page_title="CamMotionDetect Pro",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Custom title with logo
def display_title():
    st.markdown(get_title_html(), unsafe_allow_html=True)

def main():
    # Display custom title
    display_title()
    
    # Main app description
    st.markdown(get_about_card_html(), unsafe_allow_html=True)
    
    # Sidebar for parameters
    with st.sidebar:
        st.markdown("<h3>Detection Parameters</h3>", unsafe_allow_html=True)
        
        st.markdown(get_info_box_html("Adjust these parameters to fine-tune the detection sensitivity"), unsafe_allow_html=True)
        
        threshold_feature = st.slider(
            "Feature Threshold", 
            min_value=1.0, 
            max_value=50.0, 
            value=5.0,  # Lowered from 10.0 for better shake detection
            help="Lower values make the detector more sensitive to pixel differences"
        )
        
        threshold_homography = st.slider(
            "Homography Threshold", 
            min_value=5.0, 
            max_value=200.0, 
            value=15.0,  # Lowered from 100.0 for better shake detection
            help="Lower values make the detector more sensitive to camera movements"
        )
        
        min_match_count = st.slider(
            "Minimum Feature Matches", 
            min_value=4, 
            max_value=50, 
            value=8,  # Lowered from 10 for better shake detection
            help="Minimum number of matching features required for homography estimation"
        )
        
        st.markdown("<h3>How It Works</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <ol>
            <li>Upload images or video</li>
            <li>The system extracts visual features from each frame</li>
            <li>Consecutive frames are compared using feature matching</li>
            <li>A homography matrix is calculated to identify camera movement</li>
            <li>Frames with significant movement are highlighted</li>
        </ol>
        """, unsafe_allow_html=True)
    
    # Tabs for different input methods with custom styling
    tab1, tab2 = st.tabs(["📸 Image Sequence", "🎬 Video Analysis"])
    
    with tab1:
        st.markdown("<h3>Upload Image Sequence</h3>", unsafe_allow_html=True)
        
        st.markdown(get_info_box_html(
            "Upload a series of images taken from a camera. The system will analyze consecutive "
            "frames to detect significant camera movement."), unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose image files", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            frames = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                if len(frame.shape) == 3 and frame.shape[-1] == 4:  # RGBA to RGB
                    frame = frame[:, :, :3]
                frames.append(frame)
            
            st.markdown(get_info_box_html(f"Successfully loaded {len(frames)} frames."), unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button("🔍 Analyze Image Sequence", use_container_width=True)
            
            if analyze_button:
                with st.spinner("Processing images..."):
                    # Add progress bar for visual feedback
                    progress_bar = st.progress(0)
                    
                    # Process frames
                    detector = CameraMovementDetector(threshold_feature, threshold_homography, min_match_count)
                    result = detector.detect(frames)
                    
                    # Complete the progress bar
                    progress_bar.progress(100)
                    
                    # Create tabs for different views of the results
                    result_tabs = st.tabs(["📊 Analysis", "🖼️ Frames"])
                    
                    with result_tabs[0]:
                        # Plot movement scores
                        if len(result['movement_scores']) > 0:
                            st.markdown("<h3>Movement Score Analysis</h3>", unsafe_allow_html=True)
                            fig = plot_movement_scores(result)
                            st.pyplot(fig)
                        
                        # Display movement details
                        display_movement_details(result)
                    
                    with result_tabs[1]:
                        # Display all frames with movement indicators
                        display_all_frames(frames, result['movement_indices'])
    
    with tab2:
        st.markdown("<h3>Video Analysis</h3>", unsafe_allow_html=True)
        
        st.markdown(get_info_box_html(
            "Upload a video file to analyze for camera movement. The system will process the video "
            "frames and identify moments of significant camera movement."), unsafe_allow_html=True)
        
        video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        
        sample_rate = st.slider(
            "Sample Rate (process every Nth frame)", 
            min_value=1, 
            max_value=10, 
            value=1,
            help="Higher values improve performance but may miss quick movements"
        )
        
        if video_file:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            video_path = tfile.name
            tfile.close()
            
            # Display video
            st.video(video_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_video_button = st.button("🔍 Analyze Video", use_container_width=True)
            
            # Initialize session state for video analysis results
            if 'video_analysis_complete' not in st.session_state:
                st.session_state.video_analysis_complete = False
            if 'video_results' not in st.session_state:
                st.session_state.video_results = None
            
            if analyze_video_button:
                # Reset analysis state
                st.session_state.video_analysis_complete = False
                st.session_state.video_results = None
                
                # Create a placeholder for the progress
                progress_placeholder = st.empty()
                
                with progress_placeholder.container():
                    st.markdown("### 🔄 Processing Video...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # Process video and extract all frames for display
                    cap = cv2.VideoCapture(video_path)
                    all_frames = []
                    frames_for_analysis = []
                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    status_text.text(f"Extracting frames... (0/{total_frames})")
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Update progress bar
                        progress = min(100, int((frame_count / total_frames) * 50))  # 50% for frame extraction
                        progress_bar.progress(progress)
                        status_text.text(f"Extracting frames... ({frame_count}/{total_frames})")
                        
                        # Store all frames for display
                        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        # Only use every Nth frame for analysis
                        if frame_count % sample_rate == 0:
                            frames_for_analysis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        frame_count += 1
                    
                    cap.release()
                    
                    # Update progress for analysis phase
                    progress_bar.progress(50)
                    status_text.text("Analyzing camera movement...")
                    
                    # Process the sampled frames
                    detector = CameraMovementDetector(threshold_feature, threshold_homography, min_match_count)
                    result = detector.detect(frames_for_analysis)
                    
                    # Convert movement indices from sampled frames to original video frames
                    original_movement_indices = [idx * sample_rate for idx in result['movement_indices']]
                    
                    # Complete the progress bar
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Store results in session state
                    st.session_state.video_results = {
                        'all_frames': all_frames,
                        'movement_data': result,
                        'original_movement_indices': original_movement_indices
                    }
                    st.session_state.video_analysis_complete = True
                    
                    # Clear the progress placeholder after a short delay
                    import time
                    time.sleep(1)
                    progress_placeholder.empty()
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    progress_placeholder.empty()
                finally:
                    # Clean up temp file
                    if os.path.exists(video_path):
                        os.unlink(video_path)
            
            # Display results if analysis is complete
            if st.session_state.video_analysis_complete and st.session_state.video_results:
                results = st.session_state.video_results
                
                st.markdown("### ✅ Analysis Complete!")
                
                # Create tabs for different views of the results
                result_tabs = st.tabs(["📊 Analysis", "🖼️ Frames"])
                
                with result_tabs[0]:
                    # Plot movement scores
                    if len(results['movement_data']['movement_scores']) > 0:
                        st.markdown("<h3>Movement Score Analysis</h3>", unsafe_allow_html=True)
                        fig = plot_movement_scores(results['movement_data'])
                        st.pyplot(fig)
                    
                    # Display movement details (using the sampled frame indices)
                    display_movement_details(results['movement_data'])
                
                with result_tabs[1]:
                    # Display all frames with movement indicators
                    display_all_frames(results['all_frames'], results['original_movement_indices'])
                
                # Add a button to clear results and start over
                if st.button("🔄 Analyze New Video", type="secondary"):
                    st.session_state.video_analysis_complete = False
                    st.session_state.video_results = None
                    st.rerun()
    
    # Add footer
    st.markdown(get_footer_html(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
