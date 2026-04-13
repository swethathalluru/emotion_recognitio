"""
inference.py — Real-time face expression detection via webcam.
Usage:  python src/inference.py
Press Q to quit.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
import mediapipe as mp

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# ── Constants ─────────────────────────────────────────────────────────────────
EXPRESSIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

COLORS = {
    "happy":     (0,   220, 100),
    "sad":       (80,  80,  200),
    "angry":     (0,   60,  220),
    "surprised": (0,   200, 220),
    "neutral":   (160, 160, 160),
    "fear":      (180, 0,   180),
    "disgust":   (0,   180, 180),
}

# ── Landmark index groups ─────────────────────────────────────────────────────
LEFT_EYE    = [33,  160, 158, 133, 153, 144]
RIGHT_EYE   = [263, 387, 385, 362, 380, 373]
LEFT_BROW   = [70,  63,  105, 66,  107, 55,  65]
RIGHT_BROW  = [300, 293, 334, 296, 336, 285, 295]
MOUTH_INNER = [78,  95,  88,  178, 87,  14,  317, 402,
               318, 324, 308, 415, 310, 311, 312, 13,
               82,  81,  80,  191]
NOSE_TIP = 1
CHIN     = 152

# ── Feature extraction ────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_pts):
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


def mouth_aspect_ratio(mouth_pts):
    height = float(np.mean([
        np.linalg.norm(mouth_pts[2] - mouth_pts[10]),
        np.linalg.norm(mouth_pts[3] - mouth_pts[9]),
        np.linalg.norm(mouth_pts[4] - mouth_pts[8]),
    ]))
    width = float(np.linalg.norm(mouth_pts[0] - mouth_pts[6]))
    return height / (width + 1e-6)


def brow_raise_ratio(brow_pts, landmarks):
    brow_y = float(np.mean(brow_pts[:, 1]))
    nose_y = float(landmarks[NOSE_TIP, 1])
    face_h = float(np.linalg.norm(landmarks[NOSE_TIP] - landmarks[CHIN]))
    return (nose_y - brow_y) / (face_h + 1e-6)


def extract_features(landmarks: np.ndarray):
    try:
        l_eye  = landmarks[LEFT_EYE]
        r_eye  = landmarks[RIGHT_EYE]
        l_brow = landmarks[LEFT_BROW]
        r_brow = landmarks[RIGHT_BROW]
        mouth  = landmarks[MOUTH_INNER]

        ear_l  = eye_aspect_ratio(l_eye)
        ear_r  = eye_aspect_ratio(r_eye)
        mar    = mouth_aspect_ratio(mouth)
        brow_l = brow_raise_ratio(l_brow, landmarks)
        brow_r = brow_raise_ratio(r_brow, landmarks)

        mouth_w = float(np.linalg.norm(
            landmarks[MOUTH_INNER[0]] - landmarks[MOUTH_INNER[6]]
        ))
        face_w = float(np.linalg.norm(
            landmarks[LEFT_EYE[0]] - landmarks[RIGHT_EYE[3]]
        ))
        mouth_stretch = mouth_w / (face_w + 1e-6)

        return np.array([
            ear_l, ear_r, (ear_l + ear_r) / 2.0,
            mar, mouth_stretch,
            brow_l, brow_r, (brow_l + brow_r) / 2.0,
        ], dtype=np.float32)

    except Exception:
        return None

# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_label(frame, label, confidence, x=30, y=50):
    color = COLORS.get(label, (200, 200, 200))
    text  = f"{label.upper()}  {confidence:.0%}"
    # Shadow
    cv2.putText(frame, text, (x+2, y+2),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    # Text
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    # Confidence bar
    cv2.rectangle(frame, (x, y+8), (x+120, y+20), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y+8), (x+int(120*confidence), y+20), color, -1)


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:.1f}", (w-110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    # Load model
    model_path = Path(__file__).resolve().parent.parent / "models" / "expression_clf.pkl"
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("  Run first:  python src/fix_model.py")
        return

    print(f"[INFO] Loading model from {model_path}")
    clf = joblib.load(model_path)
    print("[INFO] Model loaded OK")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Webcam opened. Press Q to quit.\n")

    import time
    prev_time = time.time()

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Temporal smoother (last 7 predictions → majority vote)
    from collections import deque, Counter
    smoother = deque(maxlen=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame, skipping...")
            continue

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_lm in result.multi_face_landmarks:

                # Draw face mesh tesselation
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles
                        .get_default_face_mesh_tesselation_style(),
                )

                # Convert to numpy array
                pts = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w]
                     for lm in face_lm.landmark],
                    dtype=np.float32,
                )

                feats = extract_features(pts)
                if feats is not None:
                    feats_2d = feats.reshape(1, -1)
                    label    = clf.predict(feats_2d)[0]
                    proba    = clf.predict_proba(feats_2d).max()

                    # Smooth over last 7 frames
                    smoother.append(label)
                    stable_label = Counter(smoother).most_common(1)[0][0]

                    draw_label(frame, stable_label, proba)

        # FPS counter
        now  = time.time()
        fps  = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        cv2.imshow("Face Expression Detection  |  Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    run()