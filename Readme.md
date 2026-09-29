# Real-Time Shelf Monitoring with YOLOv8

A computer vision project that detects products on retail shelves with YOLOv8n and flags shelf sections whose detected product count is unusually low. A Streamlit app runs it on uploaded shelf images and recorded videos.

**Pipeline:** shelf image / video -> YOLOv8n product detection -> grouping into shelf rows and sections -> per-section product count -> low-count flag

## Results

### Detection accuracy

YOLOv8n fine-tuned from the COCO-pretrained `yolov8n.pt` on a subset of SKU-110K, evaluated on the SKU-110K validation split (588 images). Values are from the final epoch (30) of [`shelf_monitoring/results/results.csv`](shelf_monitoring/results/results.csv):

| Metric | Value |
|---|---:|
| Precision | 0.876 |
| Recall | 0.802 |
| mAP@50 | 0.872 |
| mAP@50-95 | 0.514 |

Unrounded values in `results.csv` (epoch 30): precision 0.87605, recall 0.80248, mAP@50 0.8716, mAP@50-95 0.51375. Epoch 30 is also the best epoch by the Ultralytics fitness score (0.1 x mAP@50 + 0.9 x mAP@50-95).

**Training setup** (from [`results/args.yaml`](shelf_monitoring/results/args.yaml)): 30 epochs, batch 16, image size 640, `fraction: 0.2` (20% of the SKU-110K training set), early-stopping patience 8 (not triggered; all 30 epochs ran); remaining settings are listed in `args.yaml`. Trained on Google Colab in about 51 minutes (3,033 s in `results.csv`).

### Inference latency (CPU)

Measured on this project's development machine, an **Intel Core i3-10110U @ 2.10 GHz** (CPU only, PyTorch 2.14.0, Ultralytics 8.4.165, 3 torch threads), with `shelf_monitoring/benchmark_fps.py` at image size 640, 5 warm-up runs and 50 timed runs on the Ultralytics sample image `bus.jpg`:

| Mean | p95 | Throughput |
|---:|---:|---:|
| 133.3 ms | 166.8 ms | 7.5 FPS |

Two consecutive passes gave a mean of 133.3-138.6 ms (7.2-7.5 FPS); the table shows the pass saved in [`results/benchmark_latency.json`](shelf_monitoring/results/benchmark_latency.json). This is **latency on one CPU, not an accuracy measurement**. The time is preprocess + inference + postprocess as reported by Ultralytics, excluding image decoding and drawing. The sample image is not a shelf photo and produced no detections, so non-maximum suppression cost is at its minimum; a dense shelf with hundreds of boxes will take longer.

## How low-count flagging works

Implemented in [`shelf_monitoring/shelf_segments.py`](shelf_monitoring/shelf_segments.py):

1. Detected boxes are sorted by vertical centre and split into **shelf rows** wherever the gap between neighbouring centres exceeds 0.6 x the median box height. Rows with fewer than 3 products are ignored.
2. Each row is divided into equal-width **sections** across the image width (default 4, adjustable in the app).
3. Products are counted per section by box centre.
4. A section is **low** when its count is below `low_ratio` x the median section count of the image (default 0.5, adjustable).
5. **Video:** a section is flagged only after it has been low in 5 consecutive processed frames, which suppresses single-frame detection dropouts.

In image mode the app draws the row/section boxes (red for low), shows a "Low-count sections" metric and a table of sections. The Confidence slider from the original app is kept.

The flagging logic is covered by synthetic-box tests (`python tests/test_shelf_segments.py`: three shelf rows, one section with missing products, asserting exactly that section is flagged, plus the 5-frame smoothing). It has not been evaluated against labelled shelf gaps.

## Limitations

- **Trained on 20% of the training set** (`fraction: 0.2`), for 30 epochs.
- **Image size 640** is small for dense, high-resolution shelf photos; small products lose detail.
- **`max_det` is 300**, which can cap the count on very dense images.
- **Count-based, not inventory-based:** it flags sections with few detections, and has no knowledge of products, planograms or stock levels. Sections are equal-width slices of the whole image, so empty space beside a shelf can look like a low section.
- **Row grouping assumes roughly level shelves**; strongly tilted or perspective-distorted photos can group rows incorrectly.
- **No live camera input** in the app: it accepts uploaded images and recorded videos only.
- Not yet checked end-to-end on real shelf photos in this repository's tests; the tests use synthetic boxes.

## Roadmap

- Train on the full SKU-110K dataset.
- Higher image size or tiled inference for dense shelves.
- Object tracker plus temporal smoothing across frames.
- ONNX / TensorRT export for faster deployment.
- Fine-tune MobileNet-SSD on the same data as a fair baseline.

## Project structure

```text
object-detection/
├── Readme.md
├── shelf_monitoring/                # main project
│   ├── app.py                       # Streamlit app
│   ├── shelf_segments.py            # row/section grouping and low-count flagging
│   ├── benchmark_fps.py             # CPU/GPU latency benchmark
│   ├── requirements.txt
│   ├── models/best.pt               # trained YOLOv8n weights
│   └── results/                     # training curves, args.yaml, results.csv, benchmark_latency.json
├── tests/
│   └── test_shelf_segments.py
└── legacy_mobilenet_demo/           # 2025 MobileNet-SSD webcam demo (VOC-pretrained, 20 classes, not trained on products)
```

## Run locally

```bash
pip install -r shelf_monitoring/requirements.txt
streamlit run shelf_monitoring/app.py       # works from any directory
python tests/test_shelf_segments.py         # logic tests, no model needed
python shelf_monitoring/benchmark_fps.py    # latency on your machine
```

Processed videos are written as H.264 (`avc1`) when the OpenCV build supports it, so the browser can play them; otherwise the app falls back to `mp4v`, shows a warning and offers a download.

## Technologies

Python · YOLOv8 (Ultralytics) · PyTorch · OpenCV · Streamlit
