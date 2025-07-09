## AI Object Detector

A real-time object detection tool built using a pre-trained deep learning model.  
Designed to detect classroom or everyday objects like pens, laptops, bottles, and more — this project is focused on fast, accurate visual recognition.

## Features

- Real-time object detection via webcam or static images
- Bounding box visualization with confidence scores
- Detects multiple objects in a single frame
- Lightweight and easy to run locally

## Tech Stack

- *Python*
- *OpenCV*
- *YOLOv8 / SSD / (Replace with your model)*
- *Ultralytics*

## How to Run

### 1. Clone the repo

git clone https://github.com/Parineeta-2307/object-detector.git
cd object-detector
2. Create a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate     # For Windows

3. Install requirements
%pip install opencv-python opencv-python-headless numpy pyttsx3

4. Run the detector
python main.py
Make sure your webcam is connected and enabled, or change the code to read from a video/image file.

Model Used
Model: YOLOv8 / SSD MobileNet / Custom-trained model
Framework: Ultralytics (YOLO) 
