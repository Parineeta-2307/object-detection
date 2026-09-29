
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import torch
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

# The model sits on CPU until its first predict(), so model.device is not a
# reliable GPU check; ask torch directly.
DEVICE = 0 if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Real-Time Shelf Monitoring",
    page_icon="📦",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# =========================================================
# HEADER
# =========================================================

st.title("📦 Real-Time Shelf Monitoring")
st.caption(
    "YOLOv8n-based retail product detection and low-count monitoring"
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

input_mode = st.sidebar.radio(
    "Input",
    ["Image", "Recorded Video"]
)

st.sidebar.divider()

st.sidebar.metric(
    "Model",
    "YOLOv8n"
)

st.sidebar.metric(
    "Dataset",
    "SKU-110K"
)

# =========================================================
# IMAGE MODE
# =========================================================

if input_mode == "Image":

    st.subheader("Shelf Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload a shelf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model.predict(
            source=image,
            imgsz=640,
            conf=confidence,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # -------------------------------------------------
        # Draw detections
        # -------------------------------------------------

        output = image.copy()

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            conf = float(box.conf[0])

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"{conf:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        count = len(result.boxes)

        # -------------------------------------------------
        # Dashboard
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Products Detected",
            count
        )

        col2.metric(
            "Confidence Threshold",
            f"{confidence:.2f}"
        )

        col3.metric(
            "Detection Status",
            "ACTIVE"
        )

        st.divider()

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            output_rgb,
            caption="YOLOv8n Product Detection",
            use_container_width=True
        )

# =========================================================
# RECORDED VIDEO MODE
# =========================================================

else:

    st.subheader("Recorded Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload a shelf video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        input_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            st.error("Could not open the uploaded video.")

        else:

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            duration = (
                frame_count / fps
                if fps > 0
                else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Video FPS",
                f"{fps:.1f}"
            )

            col2.metric(
                "Resolution",
                f"{width} × {height}"
            )

            col3.metric(
                "Duration",
                f"{duration:.1f}s"
            )

            st.divider()

            process_video = st.button(
                "▶ Process Video",
                type="primary"
            )

            if process_video:

                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps if fps > 0 else 20,
                    (width, height)
                )

                progress = st.progress(0)
                status = st.empty()

                frame_number = 0
                total_detections = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.predict(
                        source=frame,
                        imgsz=640,
                        conf=confidence,
                        device=DEVICE,
                        verbose=False
                    )

                    result = results[0]

                    output = frame.copy()

                    for box in result.boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                        conf = float(box.conf[0])

                        cv2.rectangle(
                            output,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            output,
                            f"{conf:.2f}",
                            (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

                    count = len(result.boxes)
                    total_detections += count

                    # Overlay
                    cv2.rectangle(
                        output,
                        (10, 10),
                        (280, 55),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        output,
                        f"PRODUCTS: {count}",
                        (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    writer.write(output)

                    frame_number += 1

                    if frame_count > 0:
                        progress.progress(
                            min(frame_number / frame_count, 1.0)
                        )

                    status.text(
                        f"Processing frame {frame_number}/{frame_count}"
                    )

                cap.release()
                writer.release()

                progress.progress(1.0)
                status.success(
                    f"Processed {frame_number} frames"
                )

                st.video(output_path)

                if frame_number > 0:

                    avg_count = (
                        total_detections / frame_number
                    )

                    st.metric(
                        "Average Products Detected / Frame",
                        f"{avg_count:.1f}"
                    )

                os.unlink(input_path)
