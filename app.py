import streamlit as st
import cv2
import torch
import numpy as np
import pathlib
import os
import time

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SiteGuard AI", page_icon="👷", layout="wide")
st.title("👷 SiteGuard: AI Safety Compliance System")

# --- MODEL LOADER ---
@st.cache_resource
def load_model(yolo_repo, weights_path):
    if not os.path.isdir(yolo_repo):
        raise FileNotFoundError(f"YOLO repo not found at: {yolo_repo}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")
    
    print(f"Loading model from: {weights_path}...")
    model = torch.hub.load(yolo_repo, 'custom', path=weights_path, source='local')
    return model

# --- YOUR SPECIFIC PATHS ---
# Make sure these are 100% correct on your machine
YOLO_REPO = r'your_yolo_repo_path'
WEIGHTS_PATH = r'your_weights_path'

try:
    model = load_model(YOLO_REPO, WEIGHTS_PATH)
    st.sidebar.success("System Ready: Model Loaded")
except Exception as e:
    st.sidebar.error(f"Critical Error: {e}")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
model.conf = float(conf_threshold)

st.sidebar.header("Input Source")
source_option = st.sidebar.selectbox("Select Video Source", ["Webcam (Live)", "Upload Video File", "RTSP/Local Path"])

source_path = None
stop_button = False

# Logic to handle different sources
if source_option == "Webcam (Live)":
    source_path = 0  # 0 is usually the default webcam ID
elif source_option == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save temp file
        tpath = "temp_video.mp4"
        with open(tpath, "wb") as f:
            f.write(uploaded_file.read())
        source_path = tpath
elif source_option == "RTSP/Local Path":
    source_path = st.sidebar.text_input("Enter URL or File Path:")

# --- MAIN EXECUTION ---
start_button = st.sidebar.button("Start Detection")
stop_button = st.sidebar.button("Stop")

frame_window = st.image([]) 

if start_button and source_path is not None:
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
    else:
        frame_count = 0
        # FRAME_SKIP: Process 1 out of every 3 frames for video files to match playback speed
        # Set to 1 for Webcam (no skipping), set to 3 or 4 for Video Files
        FRAME_SKIP = 5 if source_option != "Webcam (Live)" else 1 

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.write("Video Ended.")
                break

            # --- OPTIMIZATION 1: FRAME SKIPPING ---
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue # Skip this iteration to speed up processing

            # --- OPTIMIZATION 2: RESIZING ---
            # Resize frame to standard width (640px) to reduce GPU load
            # This dramatically speeds up inference
            height, width = frame.shape[:2]
            new_width = 640
            new_height = int((new_width / width) * height)
            frame = cv2.resize(frame, (new_width, new_height))

            # --- INFERENCE ---
            # YOLOv5 expects RGB, OpenCV gives BGR. 
            # We convert color later for display, but model handles BGR fine usually.
            results = model(frame)
            
            # --- COMPLIANCE LOGIC ---
            df = results.pandas().xyxy[0]
            persons = df[df['name'] == 'person']
            helmets = df[df['name'] == 'helmet']

            # Draw boxes based on logic
            for index, person in persons.iterrows():
                px1, py1, px2, py2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
                
                status = "UNSAFE"
                color = (0, 0, 255) # Red in BGR

                # Logic: Check if any helmet is "inside" the top part of the person
                for _, helmet in helmets.iterrows():
                    hx1, hy1, hx2, hy2 = int(helmet['xmin']), int(helmet['ymin']), int(helmet['xmax']), int(helmet['ymax'])
                    
                    # Check center point of helmet
                    h_cx = (hx1 + hx2) // 2
                    h_cy = (hy1 + hy2) // 2
                    
                    # If helmet center is roughly within the person's bounding box (upper half)
                    if px1 < h_cx < px2 and py1 < h_cy < (py1 + (py2 - py1) / 2):
                        status = "SAFE"
                        color = (0, 255, 0) # Green in BGR
                        break
                
                # Draw Person Box
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                # Draw Text Background for readability
                (w, h), _ = cv2.getTextSize(f"{status}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (px1, py1 - 25), (px1 + w, py1), color, -1)
                cv2.putText(frame, f"{status}", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # --- DISPLAY ---
            # Convert BGR (OpenCV) to RGB (Streamlit/Web)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")

        cap.release()