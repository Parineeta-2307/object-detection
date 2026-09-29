
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import torch
from ultralytics import YOLO

from shelf_segments import LowSectionSmoother, draw_sections, flag_low_sections

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

# The model sits on CPU until its first predict(), so model.device is not a
# reliable GPU check; ask torch directly.
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Video mode: a section is flagged only after being low in this many consecutive frames.
SMOOTHING_FRAMES = 5

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
# VIDEO WRITER
# =========================================================

def _readable(path):
    """True if OpenCV can open the file and decode its first frame."""
    cap = cv2.VideoCapture(path)
    try:
        return cap.isOpened() and cap.read()[0]
    finally:
        cap.release()


def _codec_works(codec, size):
    """Write one blank frame with this fourcc and check the file reads back."""
    probe = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    try:
        writer = cv2.VideoWriter(
            probe, cv2.VideoWriter_fourcc(*codec), 10, size
        )
        if not writer.isOpened():
            return False
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
        writer.release()
        return _readable(probe)
    finally:
        os.unlink(probe)


def make_writer(path, fps, size):
    """
    Returns (writer, codec). Prefers H.264 ('avc1'), which browsers can
    play through st.video; falls back to 'mp4v' if avc1 is unavailable
    in this OpenCV build.
    """
    for codec in ("avc1", "mp4v"):
        if _codec_works(codec, size):
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*codec), fps, size
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
    return None, None


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

n_sections = st.sidebar.slider(
    "Sections per shelf row",
    min_value=2,
    max_value=10,
    value=4,
    step=1
)

low_ratio = st.sidebar.slider(
    "Low-count ratio",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="A section is low when its count is below this fraction of the "
         "median section count."
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
        # Shelf rows / sections
        # -------------------------------------------------

        boxes = result.boxes.xyxy.cpu().numpy()

        sections = flag_low_sections(
            boxes,
            img_w=image.shape[1],
            n_sections=n_sections,
            low_ratio=low_ratio
        )

        output = draw_sections(output, sections)

        low_count = sum(s["low"] for s in sections)

        # -------------------------------------------------
        # Dashboard
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Products Detected",
            count
        )

        col2.metric(
            "Confidence Threshold",
            f"{confidence:.2f}"
        )

        col3.metric(
            "Low-count sections",
            low_count
        )

        col4.metric(
            "Shelf rows analysed",
            len({s["row"] for s in sections})
        )

        st.divider()

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            output_rgb,
            caption="YOLOv8n Product Detection "
                    "(red = low-count section, blue = normal)",
            use_container_width=True
        )

        if sections:
            st.subheader("Sections")
            st.dataframe(
                [
                    {
                        "Row": s["row"],
                        "Section": s["section"],
                        "Products": s["count"],
                        "Low": s["low"]
                    }
                    for s in sections
                ],
                hide_index=True
            )
        else:
            st.info(
                "Not enough detections to form shelf rows "
                "(a row needs at least 3 products)."
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

                writer, codec = make_writer(
                    output_path,
                    fps if fps > 0 else 20,
                    (width, height)
                )

                if writer is None:
                    st.error("No usable video codec found in this OpenCV build.")
                    st.stop()

                if codec != "avc1":
                    st.warning(
                        "H.264 (avc1) is not available in this OpenCV build, "
                        "so the output was written as mp4v. Browsers may not "
                        "be able to play it; use the download button below."
                    )

                progress = st.progress(0)
                status = st.empty()

                frame_number = 0
                total_detections = 0

                smoother = LowSectionSmoother(SMOOTHING_FRAMES)
                frames_with_flag = 0
                flagged_sections = set()

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

                    sections = smoother.update(
                        flag_low_sections(
                            result.boxes.xyxy.cpu().numpy(),
                            img_w=width,
                            n_sections=n_sections,
                            low_ratio=low_ratio
                        )
                    )

                    output = draw_sections(output, sections)

                    low_count = sum(s["low"] for s in sections)

                    if low_count:
                        frames_with_flag += 1
                        flagged_sections.update(
                            (s["row"], s["section"])
                            for s in sections if s["low"]
                        )

                    # Overlay
                    cv2.rectangle(
                        output,
                        (10, 10),
                        (400, 95),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        output,
                        f"LOW SECTIONS: {low_count}",
                        (20, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if low_count else (255, 255, 255),
                        2,
                        cv2.LINE_AA
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

                if _readable(output_path):
                    st.video(output_path)
                else:
                    st.error("The processed video could not be re-opened.")

                if codec != "avc1":
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "Download processed video",
                            f.read(),
                            file_name="shelf_output.mp4",
                            mime="video/mp4"
                        )

                if frame_number > 0:

                    avg_count = (
                        total_detections / frame_number
                    )

                    st.metric(
                        "Average Products Detected / Frame",
                        f"{avg_count:.1f}"
                    )

                    st.metric(
                        f"Frames with a flagged section "
                        f"(low for {SMOOTHING_FRAMES} consecutive frames)",
                        frames_with_flag
                    )

                    if flagged_sections:
                        st.write(
                            "Flagged at some point: "
                            + ", ".join(
                                f"R{r}S{c}"
                                for r, c in sorted(flagged_sections)
                            )
                        )

                os.unlink(input_path)
