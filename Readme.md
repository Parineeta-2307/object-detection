# Real-Time Shelf Monitoring with YOLOv8

A computer vision project for detecting products on retail shelves and flagging shelf sections with unusually low detected product counts.

## Overview

The system uses YOLOv8n trained on the SKU-110K retail shelf dataset. Detected products are grouped into shelf rows and horizontal sections, allowing simple shelf-level monitoring.

**Pipeline:**

Shelf Image / Video → YOLOv8n → Product Detection → Shelf Segmentation → Product Counting → Low-Count Flag

## Results

The model was trained on a subset of SKU-110K and evaluated on 588 validation images.

| Metric | Result |
|---|---:|
| mAP@50 | 87.20% |
| mAP@50-95 | 51.46% |
| Precision | 87.63% |
| Recall | 80.33% |

A pretrained MobileNet-SSD model was also evaluated as a baseline.

## Monitoring

The prototype:

- Detects products with bounding boxes and confidence scores.
- Groups detections into shelf rows.
- Divides rows into horizontal sections.
- Flags sections with unusually low detected product counts.
- Supports image and recorded-video testing through Streamlit.

## Project Structure

```text
object-detection/
├── app.py
├── requirements.txt
├── models/
│   └── best.pt
├── results/
└── notebooks/
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

Training used a subset of SKU-110K, and the shelf monitoring logic is based on detected product counts rather than actual inventory data. Performance can also vary with camera angle, occlusion, and hardware.

## Technologies

Python · YOLOv8 · PyTorch · OpenCV · Streamlit · ONNX Runtime
