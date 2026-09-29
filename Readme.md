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

| **Parameter**            | **Value** |
| ------------------------ | --------- |
| Model                    | YOLOv8n   |
| Dataset                  | SKU-110K  |
| Training fraction        | 20%       |
| Epochs                   | 30        |
| Batch size               | 16        |
| Image size               | 640       |
| Early stopping patience  | 8         |
| Early stopping triggered | No        |

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
| ---------------- | --------------- | -------------- |
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
```

Rows containing fewer than three detected products are ignored.

### 3. Shelf-section grouping

Each detected shelf row is divided into equal-width sections across the image.

The default configuration uses four sections, with the number adjustable through the application.

### 4. Product counting

Products are assigned to sections according to the centre point of their detected bounding boxes.

The number of detected products is then calculated independently for each section.

### 5. Low-count detection

A section is considered low-count when:

```text
section count < low_ratio × median section count
```

The default `low_ratio` is `0.5`.

This allows the system to identify shelf areas where the detected product count is substantially lower than the typical count for that image.

### 6. Temporal smoothing for video

For video input, a section is flagged only after it has remained below the threshold for five consecutive processed frames.

This reduces false alerts caused by individual-frame detection dropouts.

---

## Live and recorded video monitoring

The OpenCV-based pipeline supports both live and recorded video input.

For video processing, the system:

1. Reads frames from the input source.
2. Runs YOLOv8n object detection.
3. Draws product bounding boxes and confidence scores.
4. Groups detections into shelf rows and sections.
5. Calculates the product count for each section.
6. Applies the low-count threshold.
7. Applies five-frame temporal smoothing.
8. Overlays the resulting shelf-monitoring information on the video.

The same detection and counting logic can be applied to individual shelf images.

The Streamlit interface provides an interactive way to run the monitoring pipeline and inspect processed outputs.

---

## MobileNet-SSD baseline

A MobileNet-SSD baseline is included to provide a lightweight comparison against the YOLOv8n detector.

The comparison is intended to examine the practical trade-off between:

- Detection accuracy
- Inference speed
- Model complexity
- Suitability for constrained hardware

YOLOv8n is used as the primary detector in the final shelf-monitoring pipeline, while MobileNet-SSD provides the lightweight baseline for comparison.

---

## Limitations

- **Training data:** the YOLOv8n model was trained using 20% of the available SKU-110K training data (`fraction: 0.2`) for 30 epochs.
- **Image resolution:** image size 640 can limit detail for small products in dense, high-resolution shelf scenes.
- **Detection limit:** `max_det` is 300, which can cap detections in extremely dense images.
- **Count-based monitoring:** the system detects and counts visible products rather than determining actual inventory levels.
- **No product identity:** the current pipeline does not identify individual SKUs or distinguish between specific product types.
- **No planogram awareness:** shelf sections are defined as equal-width regions and do not use a predefined store planogram.
- **Shelf geometry:** row grouping assumes reasonably level shelves. Strong perspective distortion or highly tilted shelves can affect row assignment.
- **CPU performance:** real-time throughput depends on available hardware. The measured CPU benchmark reached approximately 7.5 FPS on an Intel Core i3-10110U.
- **Detection density:** dense shelf images containing many detections can take longer to process than the benchmark image.
- **Evaluation scope:** the automated tests use synthetic bounding boxes for validating the shelf-segmentation logic and do not constitute an end-to-end evaluation on labelled shelf gaps.

---

## Roadmap

- Train YOLOv8n on the full SKU-110K training dataset.
- Experiment with higher input resolution and tiled inference for dense shelves.
- Add object tracking for more robust temporal consistency.
- Improve temporal smoothing across longer video sequences.
- Export the detector using ONNX / TensorRT for optimized deployment.
- Extend the MobileNet-SSD comparison with additional deployment and performance metrics.
- Improve shelf geometry handling for perspective-distorted images.
- Add product/SKU-level recognition and inventory-aware monitoring.
- Evaluate the complete pipeline on labelled real-world shelf-gap scenarios.

---

## Project structure

```text
object-detection/
├── Readme.md
├── shelf_monitoring/                # Main shelf-monitoring project
│   ├── app.py                       # Streamlit application
│   ├── shelf_segments.py            # Shelf-row/section grouping and flagging
│   ├── benchmark_fps.py             # Inference latency benchmark
│   ├── requirements.txt
│   ├── models/
│   │   └── best.pt                  # Fine-tuned YOLOv8n weights
│   └── results/
│       ├── training curves
│       ├── args.yaml
│       ├── results.csv
│       └── benchmark_latency.json
│
├── tests/
│   └── test_shelf_segments.py       # Shelf segmentation and smoothing tests
│
└── legacy_mobilenet_demo/           # MobileNet-SSD baseline/demo implementation
```

---

## Running locally

Install the project dependencies:

```bash
pip install -r shelf_monitoring/requirements.txt
```

Run the Streamlit application:

```bash
streamlit run shelf_monitoring/app.py
```

Run the shelf-segmentation tests:

```bash
python tests/test_shelf_segments.py
```

Run the latency benchmark:

```bash
python shelf_monitoring/benchmark_fps.py
```

The Streamlit application can be used to process supported image and video inputs.

Processed videos are written as H.264 (`avc1`) when supported by the local OpenCV build. If H.264 encoding is unavailable, the application falls back to `mp4v`, displays a warning, and provides the processed file for download.

---

## Technologies

- Python
- YOLOv8 (Ultralytics)
- MobileNet-SSD
- PyTorch
- OpenCV
- Streamlit

---

## Key implementation details

### Detection

The primary detector is YOLOv8n fine-tuned from COCO-pretrained weights on SKU-110K.

### Shelf segmentation

Detected bounding boxes are converted into shelf rows and equal-width shelf sections using geometric relationships between detections.

### Low-count detection

Section-level product counts are compared against the median count for the image using a configurable threshold.

### Video stabilization

Five consecutive low-count observations are required before a section is flagged, reducing transient detection noise.

### Application layer

Streamlit provides the interactive interface for running the shelf-monitoring workflow, while OpenCV handles video processing and visualization.

### Baseline

MobileNet-SSD is included as a lightweight baseline for comparing the primary YOLOv8n detector in terms of accuracy and inference-speed trade-offs.

---

## Summary

This project combines fine-tuned YOLOv8n object detection with geometric shelf segmentation and temporal filtering to create a practical retail shelf-monitoring pipeline.

The system detects products, groups them into shelf sections, counts detections per section, and flags sections with unusually low detected product counts. It supports image, live-video, and recorded-video workflows and includes a MobileNet-SSD baseline for comparative evaluation.

The current YOLOv8n model achieves:

- 87.6% precision
- 80.2% recall
- 87.2% mAP@50
- 51.4% mAP@50-95
- 7.5 FPS CPU throughput on the development machine

The architecture is designed to provide a foundation for future improvements including full-dataset training, tracking, optimized inference, improved shelf geometry handling, and inventory-aware product recognition.
