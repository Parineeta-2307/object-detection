"""
Logic tests for shelf_segments using synthetic boxes (no model, no UI).
Run from the repo root:  python tests/test_shelf_segments.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shelf_monitoring"))
from shelf_segments import LowSectionSmoother, flag_low_sections, group_rows

IMG_W = 400
ROW_CENTERS_Y = [50, 150, 250]
BOX_W, BOX_H = 20, 40


def make_shelf(missing_row=None, missing_section=None, keep=0, n_sections=4, per_section=4):
    """3 shelf rows x 4 sections, `per_section` products each (xyxy boxes).
    In (missing_row, missing_section) only `keep` products remain."""
    boxes = []
    section_w = IMG_W // n_sections
    for r, cy in enumerate(ROW_CENTERS_Y):
        for s in range(n_sections):
            n = keep if (r == missing_row and s == missing_section) else per_section
            for i in range(n):
                cx = s * section_w + 10 + i * (section_w - 20) / max(per_section - 1, 1)
                boxes.append([cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2])
    return np.array(boxes, dtype=float)


def flagged(sections):
    return sorted((s["row"], s["section"]) for s in sections if s["low"])


def test_rows_are_grouped():
    rows = group_rows(make_shelf())
    assert len(rows) == 3
    assert all(len(r) == 16 for r in rows)


def test_full_shelf_flags_nothing():
    sections = flag_low_sections(make_shelf(), IMG_W)
    assert len(sections) == 12
    assert flagged(sections) == []


def test_sparse_section_is_flagged_and_only_that_one():
    sections = flag_low_sections(make_shelf(missing_row=1, missing_section=2, keep=1), IMG_W)
    assert flagged(sections) == [(1, 2)]
    (target,) = [s for s in sections if (s["row"], s["section"]) == (1, 2)]
    assert target["count"] == 1


def test_empty_section_is_flagged():
    sections = flag_low_sections(make_shelf(missing_row=1, missing_section=2, keep=0), IMG_W)
    assert flagged(sections) == [(1, 2)]


def test_no_detections():
    assert flag_low_sections(np.zeros((0, 4)), IMG_W) == []


def test_smoother_needs_five_consecutive_low_frames():
    smoother = LowSectionSmoother(n_frames=5)
    low_boxes = make_shelf(missing_row=1, missing_section=2, keep=1)
    for frame in range(1, 5):
        assert flagged(smoother.update(flag_low_sections(low_boxes, IMG_W))) == [], f"frame {frame}"
    assert flagged(smoother.update(flag_low_sections(low_boxes, IMG_W))) == [(1, 2)]  # 5th frame


def test_smoother_resets_when_section_recovers():
    smoother = LowSectionSmoother(n_frames=5)
    low_boxes = make_shelf(missing_row=1, missing_section=2, keep=1)
    for _ in range(4):
        smoother.update(flag_low_sections(low_boxes, IMG_W))
    smoother.update(flag_low_sections(make_shelf(), IMG_W))  # one normal frame breaks the streak
    for _ in range(4):
        assert flagged(smoother.update(flag_low_sections(low_boxes, IMG_W))) == []
    assert flagged(smoother.update(flag_low_sections(low_boxes, IMG_W))) == [(1, 2)]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
