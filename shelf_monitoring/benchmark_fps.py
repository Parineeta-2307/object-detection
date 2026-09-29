"""
Measure YOLOv8n inference latency on this machine (latency only, not accuracy).

Usage (from any directory):
    python benchmark_fps.py [image_or_folder] [--runs 50] [--imgsz 640]

With no path it uses the sample image shipped with ultralytics (assets/bus.jpg).
Reported time per image = preprocess + inference + postprocess (NMS), as measured
by ultralytics (result.speed); image decoding and drawing are not included.
"""
import argparse
import glob
import json
import platform
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "best.pt"
DEFAULT_IMAGE = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


def cpu_name():
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_paths(source):
    source = Path(source)
    if source.is_dir():
        paths = sorted(glob.glob(str(source / "*.jpg")))
        if not paths:
            raise SystemExit(f"No .jpg files in {source}")
        return paths
    return [str(source)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(DEFAULT_IMAGE))
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(str(MODEL_PATH))
    device = 0 if torch.cuda.is_available() else "cpu"
    paths = (collect_paths(args.source) * args.runs)[: args.runs]

    for _ in range(5):  # warmup
        model.predict(paths[0], imgsz=args.imgsz, device=device, verbose=False)

    times, n_det = [], []
    for p in paths:
        r = model.predict(p, imgsz=args.imgsz, device=device, verbose=False)[0]
        times.append(sum(r.speed.values()))  # preprocess + inference + postprocess (ms)
        n_det.append(len(r.boxes))

    t = np.array(times)
    out = {
        "cpu": cpu_name(),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "imgsz": args.imgsz,
        "source": args.source,
        "runs": len(t),
        "mean_ms": round(float(t.mean()), 1),
        "p95_ms": round(float(np.percentile(t, 95)), 1),
        "fps": round(float(1000 / t.mean()), 1),
        "mean_detections_per_image": round(float(np.mean(n_det)), 1),
    }
    print(f"cpu={out['cpu']}  device={device}  imgsz={args.imgsz}  runs={len(t)}")
    print(f"mean={t.mean():.1f} ms  p95={np.percentile(t, 95):.1f} ms  FPS={1000 / t.mean():.1f}  "
          f"(mean detections/image: {out['mean_detections_per_image']})")
    out_path = HERE / "results" / "benchmark_latency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
