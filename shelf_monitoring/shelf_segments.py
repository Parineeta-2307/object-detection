"""
Shelf-level low-count flagging for shelf_monitoring/app.py.
Groups detections into shelf rows by vertical position, splits each row into
equal-width sections, counts products per section, and flags sections whose
count is well below the typical section count.
"""
import numpy as np
import cv2


def group_rows(boxes, gap_factor=0.6):
    """boxes: (N, 4) xyxy array. Returns list of index lists, one per shelf row."""
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    gap = gap_factor * np.median(heights)
    order = np.argsort(cy)
    rows, current = [], [order[0]]
    for i in order[1:]:
        if cy[i] - cy[current[-1]] > gap:
            rows.append(current)
            current = [i]
        else:
            current.append(i)
    rows.append(current)
    return rows


def flag_low_sections(boxes, img_w, n_sections=4, low_ratio=0.5, min_row_items=3):
    """Returns list of dicts: row, section, count, x1, x2, y1, y2, low (bool)."""
    sections = []
    for r, idx in enumerate(group_rows(boxes)):
        if len(idx) < min_row_items:
            continue
        rb = boxes[idx]
        y1, y2 = int(rb[:, 1].min()), int(rb[:, 3].max())
        cx = (rb[:, 0] + rb[:, 2]) / 2
        edges = np.linspace(0, img_w, n_sections + 1)
        for s in range(n_sections):
            count = int(((cx >= edges[s]) & (cx < edges[s + 1])).sum())
            sections.append({"row": r, "section": s, "count": count,
                             "x1": int(edges[s]), "x2": int(edges[s + 1]), "y1": y1, "y2": y2})
    if sections:
        typical = np.median([s["count"] for s in sections])
        for s in sections:
            s["low"] = bool(s["count"] < low_ratio * typical)
    return sections


def draw_sections(img, sections):
    for s in sections:
        color = (0, 0, 255) if s["low"] else (255, 200, 0)
        cv2.rectangle(img, (s["x1"], s["y1"]), (s["x2"], s["y2"]), color, 2)
        cv2.putText(img, f"R{s['row']}S{s['section']}:{s['count']}", (s["x1"] + 5, s["y1"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


class LowSectionSmoother:
    """
    Temporal smoothing for video: a section is reported as low only after it
    has been low in `n_frames` consecutive processed frames. Sections are keyed
    by (row, section), so a change in the number of detected rows between
    frames resets the affected streaks.
    """

    def __init__(self, n_frames=5):
        self.n_frames = n_frames
        self.streaks = {}

    def update(self, sections):
        """Adds "low_raw" (this frame only) and overwrites "low" with the smoothed flag."""
        seen = set()
        for s in sections:
            key = (s["row"], s["section"])
            seen.add(key)
            s["low_raw"] = s["low"]
            self.streaks[key] = self.streaks.get(key, 0) + 1 if s["low_raw"] else 0
            s["low"] = self.streaks[key] >= self.n_frames
        for key in list(self.streaks):
            if key not in seen:
                del self.streaks[key]
        return sections
