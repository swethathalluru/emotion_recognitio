import numpy as np

# Landmark index groups (MediaPipe 468-point mesh)
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
               324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
               310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE   = [263, 387, 385, 362, 380, 373]
LEFT_BROW   = [70, 63, 105, 66, 107, 55, 65]
RIGHT_BROW  = [300, 293, 334, 296, 336, 285, 295]
NOSE_TIP    = 1
CHIN        = 152

def eye_aspect_ratio(eye_pts):
    """EAR — blink / squint detection."""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(inner_pts):
    """MAR — open/smile width ratio."""
    height = np.mean([
        np.linalg.norm(inner_pts[2] - inner_pts[10]),
        np.linalg.norm(inner_pts[3] - inner_pts[9]),
        np.linalg.norm(inner_pts[4] - inner_pts[8]),
    ])
    width = np.linalg.norm(inner_pts[0] - inner_pts[6])
    return height / (width + 1e-6)

def brow_raise_ratio(brow_pts, face_pts, nose_idx=NOSE_TIP):
    """Brow height relative to face height — surprise / anger."""
    brow_y = np.mean(brow_pts[:, 1])
    nose_y = face_pts[nose_idx, 1]
    face_h = np.linalg.norm(face_pts[NOSE_TIP] - face_pts[CHIN])
    return (nose_y - brow_y) / (face_h + 1e-6)

def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Return a 1-D feature vector for one face."""
    l_eye  = landmarks[LEFT_EYE]
    r_eye  = landmarks[RIGHT_EYE]
    l_brow = landmarks[LEFT_BROW]
    r_brow = landmarks[RIGHT_BROW]
    mouth  = landmarks[MOUTH_INNER]

    ear_l = eye_aspect_ratio(l_eye)
    ear_r = eye_aspect_ratio(r_eye)
    mar   = mouth_aspect_ratio(mouth)
    brow_l = brow_raise_ratio(l_brow, landmarks)
    brow_r = brow_raise_ratio(r_brow, landmarks)

    # Mouth corner pull (smile width normalized)
    mouth_w = np.linalg.norm(
        landmarks[MOUTH_INNER[0]] - landmarks[MOUTH_INNER[6]]
    )
    face_w = np.linalg.norm(landmarks[LEFT_EYE[0]] - landmarks[RIGHT_EYE[3]])
    mouth_stretch = mouth_w / (face_w + 1e-6)

    return np.array([
        ear_l, ear_r, (ear_l + ear_r) / 2,
        mar, mouth_stretch,
        brow_l, brow_r, (brow_l + brow_r) / 2,
    ], dtype=np.float32)