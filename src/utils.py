"""
utils.py — Shared helpers for the face expression detection pipeline.

Covers:
  - Logging setup
  - Directory / path helpers
  - Landmark normalization (pose-invariant)
  - Temporal smoother for real-time inference
  - Confusion-matrix + per-class metric reporting
  - FER2013 CSV → image-file converter
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections import deque, Counter
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str = "face_expression", level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger()

# ── Expressions ───────────────────────────────────────────────────────────────

EXPRESSIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# BGR colours for cv2 overlays
EXPRESSION_COLORS: dict[str, tuple[int, int, int]] = {
    "happy":     (0,   220, 100),
    "sad":       (200, 80,  80),
    "angry":     (0,   60,  220),
    "surprised": (0,   200, 220),
    "neutral":   (160, 160, 160),
    "fear":      (180, 0,   180),
    "disgust":   (0,   180, 180),
}

# ── Path helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it doesn't exist. Returns a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def data_dir(subdir: str = "") -> Path:
    root = project_root() / "data"
    return ensure_dir(root / subdir) if subdir else root


def models_dir() -> Path:
    return ensure_dir(project_root() / "models")

# ── Landmark normalisation ────────────────────────────────────────────────────

# Indices of the inner eye corners (used as the reference baseline)
_LEFT_INNER_CORNER  = 133   # MediaPipe FaceMesh
_RIGHT_INNER_CORNER = 362

def normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make raw (x, y, z) landmarks pose-invariant.

    Steps
    -----
    1. Translate so that the midpoint of the two inner eye corners is at origin.
    2. Scale so that the inter-ocular distance (IOD) == 1.0.

    Parameters
    ----------
    landmarks : np.ndarray, shape (468, 3)
        Raw pixel-space landmarks from MediaPipe (x in [0,W], y in [0,H]).

    Returns
    -------
    np.ndarray, shape (468, 3)  — normalised landmarks
    """
    left  = landmarks[_LEFT_INNER_CORNER]
    right = landmarks[_RIGHT_INNER_CORNER]

    centroid = (left + right) / 2.0
    iod = float(np.linalg.norm(right - left))
    if iod < 1e-6:
        return landmarks  # degenerate face — return as-is

    return (landmarks - centroid) / iod


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Return landmarks as a flat float32 vector (468*3 = 1404 dims)."""
    return landmarks.astype(np.float32).ravel()

# ── Temporal smoother ─────────────────────────────────────────────────────────

class TemporalSmoother:
    """
    Keep a rolling window of predictions; return the majority-vote label.

    Parameters
    ----------
    window : int
        Number of frames to smooth over (default 7).
    """

    def __init__(self, window: int = 7) -> None:
        self._buf: deque[str] = deque(maxlen=window)

    def update(self, label: str) -> str:
        self._buf.append(label)
        return Counter(self._buf).most_common(1)[0][0]

    def reset(self) -> None:
        self._buf.clear()

    @property
    def stable(self) -> bool:
        """True once the buffer is full."""
        return len(self._buf) == self._buf.maxlen

# ── FPS counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)
        self._prev = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._prev)
        self._prev = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_label(
    frame: np.ndarray,
    label: str,
    confidence: float,
    position: tuple[int, int] = (30, 50),
) -> None:
    """
    Render *label* + confidence bar on *frame* in-place.

    Parameters
    ----------
    frame      : BGR image (modified in place).
    label      : Expression string, e.g. ``"happy"``.
    confidence : Float in [0, 1].
    position   : Top-left corner (x, y) for the text.
    """
    color = EXPRESSION_COLORS.get(label, (200, 200, 200))
    x, y = position
    text = f"{label.capitalize()}  {confidence:.0%}"

    # Shadow for legibility on any background
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    # Confidence bar (100 px wide)
    bar_x, bar_y, bar_h = x, y + 12, 10
    bar_w = 100
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * confidence), bar_y + bar_h),
                  color, -1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:.1f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_report_custom(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str] | None = None,
) -> str:
    """
    Return a human-readable per-class precision / recall / F1 table.

    Unlike sklearn's version this has no external dependencies beyond numpy.
    """
    labels = labels or sorted(set(y_true) | set(y_pred))
    lines = [
        f"{'Label':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}",
        "-" * 42,
    ]
    macro_p = macro_r = macro_f = 0.0
    for lbl in labels:
        tp = sum(t == lbl and p == lbl for t, p in zip(y_true, y_pred))
        fp = sum(t != lbl and p == lbl for t, p in zip(y_true, y_pred))
        fn = sum(t == lbl and p != lbl for t, p in zip(y_true, y_pred))
        sup = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        macro_p += prec; macro_r += rec; macro_f += f1
        lines.append(f"{lbl:<12} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {sup:>6d}")

    n = len(labels)
    lines += [
        "-" * 42,
        f"{'macro avg':<12} {macro_p/n:>6.3f} {macro_r/n:>6.3f} {macro_f/n:>6.3f}",
    ]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    lines.append(f"\nAccuracy: {acc:.4f}  ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")
    return "\n".join(lines)


def confusion_matrix_str(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str] | None = None,
) -> str:
    """Return an ASCII confusion matrix."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    mat = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1

    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    lines = ["Confusion matrix (rows=true, cols=pred)", header]
    for i, lbl in enumerate(labels):
        row = f"{lbl:<{col_w}}" + "".join(f"{mat[i,j]:>{col_w}}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)

# ── FER2013 CSV → image files ─────────────────────────────────────────────────

FER_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]


def fer2013_csv_to_images(
    csv_path: str | Path,
    output_dir: str | Path,
    split: str = "Training",
) -> int:
    """
    Convert the official FER2013 ``fer2013.csv`` file to per-class JPEG images.

    Parameters
    ----------
    csv_path   : Path to ``fer2013.csv``.
    output_dir : Root directory; images are written to ``output_dir/<label>/``.
    split      : One of ``"Training"``, ``"PublicTest"``, ``"PrivateTest"``.

    Returns
    -------
    int  Number of images written.
    """
    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"FER2013 CSV not found: {csv_path}")

    written = 0
    counters: dict[str, int] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Usage", "") != split:
                continue

            label_idx = int(row["emotion"])
            label     = FER_LABELS[label_idx]
            pixels    = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
            img       = pixels.reshape(48, 48)

            out_dir = ensure_dir(output_dir / label)
            idx     = counters.get(label, 0)
            out_path = out_dir / f"{label}_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), img)

            counters[label] = idx + 1
            written += 1

            if written % 1000 == 0:
                log.info("Converted %d images...", written)

    log.info("Done. Wrote %d images to %s", written, output_dir)
    return written

# ── Video / frame helpers ─────────────────────────────────────────────────────

def resize_keep_aspect(
    frame: np.ndarray,
    max_side: int = 640,
) -> np.ndarray:
    """Downscale *frame* so its longest side is at most *max_side*."""
    h, w = frame.shape[:2]
    scale = min(max_side / w, max_side / h, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB (MediaPipe requirement)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)