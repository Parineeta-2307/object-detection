# AI Object Detection System

A real-time object detection system built for retail shelf monitoring, using YOLO v11 and TensorFlow. The pipeline handles live video input, runs inference frame-by-frame, and surfaces detections through an OpenCV-based visual dashboard.

---

## Tech Stack

- **Python** - core language
- **YOLOv11** - object detection model
- **TensorFlow** - model loading and inference backend
- **OpenCV** - video capture, frame preprocessing, and bounding box rendering

---

## Features

- Real-time object detection on live video or uploaded footage
- Bounding box rendering with class labels and confidence scores
- Retail shelf monitoring use case — detects product presence, misplacement, and gaps
- Optimized inference pipeline with reduced per-frame processing time

---

## Project Structure

```
ai-object-detection/
├── model/
│   └── yolov11_weights.pt       # Pretrained YOLO v11 weights
├── src/
│   ├── detect.py                # Core inference script
│   ├── preprocess.py            # Frame normalization and resizing
│   └── visualize.py             # Bounding box and label rendering
├── samples/
│   └── shelf_sample.mp4         # Sample input video
├── requirements.txt
└── README.md
```

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/Parineeta-2307/object-detection.git
cd object-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run detection on a sample video**
```bash
python src/detect.py --source samples/shelf_sample.mp4
```

**4. Run on live webcam feed**
```bash
python src/detect.py --source 0
```

---

## How It Works

1. **Input** - Video frames are captured from a file or live camera feed using OpenCV.
2. **Preprocessing** - Each frame is resized and normalized to match the input format expected by the YOLO v11 model.
3. **Inference** - The preprocessed frame is passed through the YOLO v11 model, which returns bounding box coordinates, class labels, and confidence scores.
4. **Visualization** - Detections are rendered back onto the frame with labeled bounding boxes and displayed in real time.

---

## Key Challenge

During development, bounding box outputs were flickering inconsistently across frames despite identical input objects. The root cause was a normalization mismatch - the preprocessing pipeline applied a different pixel scaling than what the model expected at inference time. Aligning the normalization step to match the model's training configuration resolved the instability entirely.

---

## Requirements

```
tensorflow>=2.10
opencv-python>=4.7
ultralytics>=8.0
numpy
```

---

## Future Improvements

- Add support for multi-camera input streams
- Integrate an alert system for out-of-stock shelf detection
- Export detection logs to CSV for downstream analytics
- Build a lightweight web dashboard using Streamlit

---

## Author

**Parineeta Rana**  
[GitHub](https://github.com/Parineeta-2307) · [LinkedIn](https://www.linkedin.com/in/parineeta-rana/)
