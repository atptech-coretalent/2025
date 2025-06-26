import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64
from typing import Dict, Any
from movement_detector.scoring import classify_movement_type

def plot_movement_scores(movement_data: Dict[str, Any]) -> plt.Figure:
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#f0f2f6')
    scores = movement_data['movement_scores']
    frames = list(range(len(scores)))
    ax.plot(frames, scores, color='#1e3d59', linewidth=2, alpha=0.7)
    ax.fill_between(frames, scores, color='#1e3d59', alpha=0.1)
    movement_indices = movement_data['movement_indices']
    if movement_indices:
        movement_scores = [scores[i] if i < len(scores) else 0 for i in movement_indices]
        ax.scatter(movement_indices, movement_scores, color='#ff6e40', s=100, zorder=5, \
                  label='Movement Detected', edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Movement Score', fontsize=12, fontweight='bold')
    ax.set_title('Camera Movement Detection Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    if movement_indices and len(scores) > 0:
        threshold = min([scores[i] for i in movement_indices]) if movement_indices else 0
        ax.axhline(y=threshold, color='#ff6e40', linestyle='--', alpha=0.7, \
                  label=f'Threshold: {threshold:.2f}')
    ax.legend(loc='upper right', frameon=True, facecolor='white')
    plt.tight_layout()
    return fig

def display_movement_details(movement_data: Dict[str, Any]) -> None:
    movement_indices = movement_data['movement_indices']
    if not movement_indices:
        st.markdown("""
        <div class="info-box">
            <h3>Analysis Result</h3>
            <p>No significant camera movement detected in the analyzed frames.</p>
        </div>
        """, unsafe_allow_html=True)
        total_frames = len(movement_data.get('movement_scores', []))
        if total_frames > 0:
            report_text = create_simple_report(movement_data, total_frames)
            st.download_button(
                label="📄 Download Summary Report",
                data=report_text,
                file_name="movement_report_no_movement.txt",
                mime="text/plain",
                help="Download a simple summary of the movement detection results"
            )
        return
    st.markdown(f"""
    <div class="success-box">
        <h3>Analysis Result</h3>
        <p>Significant camera movement detected in {len(movement_indices)} frames!</p>
        <p>Movement detected at frames: {', '.join(map(str, movement_indices))}</p>
    </div>
    """, unsafe_allow_html=True)
    total_frames = len(movement_data.get('movement_scores', []))
    if total_frames > 0:
        report_text = create_simple_report(movement_data, total_frames)
        st.download_button(
            label="📄 Download Summary Report",
            data=report_text,
            file_name=f"movement_report_{len(movement_indices)}_frames.txt",
            mime="text/plain",
            help="Download a simple summary of the movement detection results"
        )
    if 'transformation_matrices' in movement_data:
        movement_types = []
        for idx in movement_indices:
            if idx < len(movement_data['transformation_matrices']):
                matrix = movement_data['transformation_matrices'][idx]
                if matrix is not None:
                    movement_type = classify_movement_type(matrix)
                    movement_types.append((idx, movement_type))
        if movement_types:
            st.markdown("<h3>Movement Analysis</h3>", unsafe_allow_html=True)
            for idx, movement_type in enumerate(movement_types):
                frame_idx, movement_data = movement_type
                movement_values = {
                    'Translation': abs(movement_data['translation_x']) + abs(movement_data['translation_y']),
                    'Rotation': abs(movement_data['rotation']),
                    'Scaling': abs(movement_data['scaling'] - 1.0) * 100,
                    'Perspective': movement_data['perspective'] * 10
                }
                dominant_type = max(movement_values, key=lambda k: movement_values[k])
                with st.expander(f"Frame {frame_idx} - Dominant Movement: {dominant_type}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                            <h4>Translation</h4>
                            <p>X-axis: {movement_data['translation_x']:.2f} pixels</p>
                            <p>Y-axis: {movement_data['translation_y']:.2f} pixels</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
                            <h4>Rotation</h4>
                            <p>{movement_data['rotation']:.2f} degrees</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                            <h4>Scaling</h4>
                            <p>{movement_data['scaling']:.2f}x</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
                            <h4>Perspective Distortion</h4>
                            <p>{movement_data['perspective']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("<h4>Movement Visualization</h4>", unsafe_allow_html=True)
                    tx = movement_data['translation_x']
                    ty = movement_data['translation_y']
                    viz_size = 200
                    viz = np.ones((viz_size, viz_size, 3), dtype=np.uint8) * 255
                    center = (viz_size // 2, viz_size // 2)
                    scale = 50 / max(1, max(abs(tx), abs(ty)))
                    end_point = (
                        int(center[0] + tx * scale),
                        int(center[1] + ty * scale)
                    )
                    cv2.arrowedLine(viz, center, end_point, (30, 61, 89), 3, tipLength=0.2)
                    cv2.circle(viz, center, 5, (255, 110, 64), -1)
                    cv2.putText(viz, "Start", (center[0] - 30, center[1] - 10), \
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    st.image(viz, use_column_width=False, width=200)

def display_all_frames(frames, movement_indices=None):
    st.markdown("<h3>Movement Detected Frames</h3>", unsafe_allow_html=True)
    if movement_indices and len(movement_indices) > 0:
        movement_frames = [(idx, frames[idx]) for idx in movement_indices if idx < len(frames)]
        st.markdown(f"""
        <div class="info-box">
            <p>Displaying {len(movement_frames)} frames with detected camera movement (out of {len(frames)} total frames)</p>
        </div>
        """, unsafe_allow_html=True)
        num_cols = 4
        st.markdown("<div class='frame-container'>", unsafe_allow_html=True)
        for row_start in range(0, len(movement_frames), num_cols):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                frame_list_idx = row_start + col_idx
                if frame_list_idx < len(movement_frames):
                    original_frame_idx, frame = movement_frames[frame_list_idx]
                    display_frame = frame.copy()
                    h, w = display_frame.shape[:2]
                    cv2.rectangle(display_frame, (0, 0), (w-1, h-1), (255, 110, 64), 5)
                    cv2.putText(display_frame, "MOVEMENT", (10, 30), \
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 110, 64), 2)
                    frame_class = "movement-frame"
                    caption_class = "movement-caption"
                    caption = f"Frame {original_frame_idx} - MOVEMENT DETECTED"
                    cols[col_idx].markdown(f"""
                    <div class="{frame_class}">
                        <img src="data:image/png;base64,{image_to_base64(display_frame)}" style="width: 100%;">
                        <div class="frame-caption {caption_class}">{caption}</div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            <p>No camera movement detected in the analyzed video ({len(frames)} frames analyzed)</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
            <h4>💡 Suggestions:</h4>
            <ul>
                <li>Try lowering the <strong>Homography Threshold</strong> in the sidebar (current value might be too high)</li>
                <li>Try lowering the <strong>Feature Threshold</strong> for more sensitive detection</li>
                <li>Reduce the <strong>Minimum Feature Matches</strong> requirement</li>
                <li>Check if your video actually contains camera movement (not just object movement)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def create_simple_report(movement_data: Dict[str, Any], total_frames: int) -> str:
    from datetime import datetime
    movement_indices = movement_data['movement_indices']
    report = []
    report.append("# Camera Movement Detection Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Summary")
    report.append(f"Total frames analyzed: {total_frames}")
    report.append(f"Movement detected frames: {len(movement_indices)}")
    if movement_indices:
        movement_percentage = (len(movement_indices) / total_frames) * 100
        report.append(f"Movement percentage: {movement_percentage:.1f}%")
        report.append("Status: ✅ MOVEMENT DETECTED")
        report.append("")
        report.append("## Movement Frames")
        report.append(f"Frames with movement: {', '.join(map(str, movement_indices))}")
    else:
        report.append("Movement percentage: 0.0%")
        report.append("Status: ❌ NO MOVEMENT DETECTED")
    report.append("")
    report.append("---")
    report.append("Report generated by CamMotionDetect Pro")
    return "\n".join(report)

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_custom_css() -> str:
    return """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-container {
        background-color: #1e3d59;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .subtitle {
        color: #ffc13b;
        font-style: italic;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #212529;
        border: 1px solid #dee2e6;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 5px solid #007bff;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: #212529;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: #155724;
        font-weight: 500;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin-bottom: 1rem;
        color: #856404;
        font-weight: 500;
    }
    .frame-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .movement-frame {
        border: 4px solid #ff5722;
        border-radius: 8px;
    }
    .normal-frame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    .frame-caption {
        text-align: center;
        padding: 0.5rem;
        font-weight: bold;
        color: #495057;
        background-color: #f8f9fa;
        border-top: 1px solid #dee2e6;
    }
    .movement-caption {
        color: #dc3545;
        background-color: #f8d7da;
        font-weight: 600;
    }
    .pagination-controls {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #666;
    }
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6e40;
        color: white;
    }
    </style>
    """

def get_title_html() -> str:
    return """
    <div class="title-container">
        <h1>CamMotionDetect Pro</h1>
        <p class="subtitle">Advanced Camera Movement Detection System</p>
    </div>
    """

def get_about_card_html() -> str:
    return """
    <div class="card">
        <h3>About This Tool</h3>
        <p>This advanced system detects significant camera movement in image sequences or videos. 
        Unlike basic motion detection that focuses on moving objects within a scene, 
        CamMotionDetect Pro identifies when the entire camera has been moved, tilted, or shifted.</p>
        <p>The system uses sophisticated computer vision techniques including feature matching, 
        homography estimation, and transformation analysis to distinguish between object movement 
        and camera movement.</p>
    </div>
    """

def get_info_box_html(text: str) -> str:
    return f"""
    <div class="info-box">
        <p>{text}</p>
    </div>
    """

def get_footer_html() -> str:
    return """
    <div class="footer">
        <p>CamMotionDetect Pro | Advanced Camera Movement Detection System</p>
        <p>Developed by ATP Core Talent 2025 Candidate</p>
    </div>
    """ 