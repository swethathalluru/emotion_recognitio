import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

class LandmarkExtractor:
    def __init__(self, static_mode=False, max_faces=1, min_confidence=0.5):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,          # enables iris + detailed lips
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )

    def extract(self, frame: np.ndarray) -> list[np.ndarray] | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        all_faces = []
        for face_landmarks in results.multi_face_landmarks:
            pts = np.array([
                [lm.x * w, lm.y * h, lm.z * w]   # z scaled by image width
                for lm in face_landmarks.landmark
            ], dtype=np.float32)
            all_faces.append(pts)
        return all_faces