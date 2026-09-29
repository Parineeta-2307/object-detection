# Real-Time Shelf Monitoring with YOLOv8

A computer vision project for real-time retail shelf monitoring using YOLOv8n to detect products and identify shelf sections with unusually low product counts.

The system combines YOLOv8n-based object detection with an OpenCV processing pipeline to analyze shelf images, live video, and recorded video input. Detected products are grouped into shelf rows and sections, after which sections with unusually low product counts are flagged.

A MobileNet-SSD baseline is also included for comparative evaluation of detection accuracy and inference speed.

**Pipeline:** shelf image / live video / recorded video -> YOLOv8n product detection -> bounding box and confidence overlay -> shelf-row grouping -> shelf-section grouping -> per-section product count -> low-count flagging

---

## Results

### Detection accuracy

YOLOv8n was fine-tuned from the COCO-pretrained `yolov8n.pt` on a subset of SKU-110K and evaluated on the SKU-110K validation split (588 images).

Values are from the final epoch (30) of [`shelf_monitoring/results/results.csv`](https://github.com/Parineeta-2307/object-detection/blob/main/shelf_monitoring/results/results.csv):

| **Metric** | **YOLOv8n** |
| ---------- | ----------- |
| Precision  | 0.876       |
| Recall     | 0.802       |
| mAP@50     | 0.872       |
| mAP@50-95  | 0.514       |

Unrounded values in `results.csv` (epoch 30):

- Precision: `0.87605`
- Recall: `0.80248`
- mAP@50: `0.8716`
- mAP@50-95: `0.51375`

Epoch 30 is also the best epoch by the Ultralytics fitness score.

The project additionally uses a MobileNet-SSD model as a lightweight baseline for the retail object-detection workflow. The baseline provides a reference point for comparing the primary YOLOv8n detector across detection accuracy and inference speed.

### Training setup

Training configuration is recorded in [`results/args.yaml`](https://github.com/Parineeta-2307/object-detection/blob/main/shelf_monitoring/results/args.yaml).

| **Parameter** | **Value** |
|---------------|-----------|
| Model | YOLOv8n |
| Dataset | SKU-110K |
| Training fraction | 20% |
| Epochs | 30 |
| Batch size | 16 |
| Image size | 640 |
| Early stopping patience | 8 |
| Early stopping triggered | No |

The model was fine-tuned using Kaggle GPU notebooks in approximately 51 minutes (`3,033 s` recorded in `results.csv`).

---

## Inference latency

YOLOv8n inference was benchmarked on the project's development machine:

**CPU:** Intel Core i3-10110U @ 2.10 GHz  
**Device:** CPU only  
**PyTorch:** 2.14.0  
**Ultralytics:** 8.4.165  
**Torch threads:** 3  
**Image size:** 640  
**Warm-up runs:** 5  
**Timed runs:** 50

The benchmark uses `shelf_monitoring/benchmark_fps.py` with the Ultralytics sample image `bus.jpg`.

| **Mean latency** | **p95 latency** | **Throughput** |
|------------------|-----------------|----------------|
| 133.3 ms         | 166.8 ms        | 7.5 FPS        |

Two consecutive passes produced a mean latency of approximately `133.3-138.6 ms`, corresponding to `7.2-7.5 FPS`.

The recorded benchmark is available in [`results/benchmark_latency.json`](https://github.com/Parineeta-2307/object-detection/blob/main/shelf_monitoring/results/benchmark_latency.json).

This benchmark measures CPU latency rather than detection accuracy. The reported time includes preprocessing, inference, and postprocessing as reported by Ultralytics, while excluding image decoding and drawing.

The benchmark image is not a shelf image and produced no detections. Therefore, non-maximum suppression was measured under a minimal detection load; dense shelf scenes containing hundreds of detections can require additional processing time.

The MobileNet-SSD baseline is used as the lightweight comparison point for evaluating the practical accuracy-versus-speed trade-off.

---

## How the shelf monitoring works

The shelf-monitoring logic is implemented in [`shelf_monitoring/shelf_segments.py`](https://github.com/Parineeta-2307/object-detection/blob/main/shelf_monitoring/shelf_segments.py).

### 1. Product detection

YOLOv8n detects products and produces bounding boxes with confidence scores.

The detections are used as the input to the shelf segmentation and counting pipeline.

### 2. Shelf-row grouping

Detected bounding boxes are sorted by their vertical centre.

Neighbouring detections are grouped into the same shelf row unless the vertical gap between them exceeds:

```text
0.6 × median bounding-box height
