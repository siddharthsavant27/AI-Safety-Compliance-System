## SiteGuard: An AI-Driven Industrial Safety Compliance System

SiteGuard is a computer vision system that provides a real-time solution for detecting compliance by workers with wearing Personal Protection Equipment (PPE) while on construction sites. SiteGuard's technology consists of a fine-tuned instance of the YOLOv5 object detection model in order to detect workers and ensure that they are wearing PPE (hard hats and safety vests) during all work activity.

In contrast to the typical approaches to object detection, which simply register the presence of PPE on the construction site, SiteGuard provides an association between each detected worker and their respective piece of PPE using geometric logic to label workers as "SAFE" (green) or "UNSAFE" (red) based on their level of PPE compliance.

## Key Features

- Real-Time Inference - Detects and labels gear at 30+ FPS using CUDA.
- Custom Safety Logic - Uses geometric logic to determine whether a detected helmet belongs to a detected person.
- Interactive Dashboard - SiteGuard has an interactive web application (Streamlit) that allows users to upload their own video files, set the desired confidence levels for detections of PPE on workers, and generate processed safety reports.
- Robust - Designed to handle high-resolution video files using frame skipping and resizing techniques.

## Technical Stack

- Language: Python 3.11
- Computer Vision: OpenCV; Ultralytics YOLOv5
- Deep Learning: PyTorch; GPU enabled
- Interface: Streamlit
- Data Handling: Pandas; NumPy

## Performance Metrics

- Model: YOLOv5s (fine-tuned on >200 labeled images)
- mAP@0.5 = 0.85
- Precision (helmet detection) = 94%
- Recall (person detection) = 91%

## 📂 Project Structure
```text
├── app.py                 # Main Streamlit Dashboard application
├── requirements.txt       # Python dependencies
├── demo.gif               # Demo for README
├── yolov5/                # YOLOv5 submodule
└── README.md              # Project Documentation
