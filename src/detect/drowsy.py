from typing import List, Tuple, Optional
import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(pts_xy: np.ndarray, idxs: List[int]) -> float:
    p = pts_xy[idxs]
    horiz = np.linalg.norm(p[0] - p[3]) + 1e-6
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    return float(0.5 * (v1 + v2) / horiz)

class IRDrowsyDetector:
    """
    - IR 그레이 프레임 입력
    - Face bbox (mesh 전체 min/max)
    - 눈 랜드마크 + EAR 계산
    - 알람 텍스트 오버레이
    """
    def __init__(self):
        self.fm = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def process(self, gray: np.ndarray, ear_thresh: float) -> Tuple[Optional[float], np.ndarray]:
        if gray is None:
            return None, None
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        if not res.multi_face_landmarks:
            return None, bgr

        h, w = gray.shape[:2]
        pts = np.array([(p.x, p.y) for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)

        l_ear = eye_aspect_ratio(pts, LEFT_EYE_IDXS)
        r_ear = eye_aspect_ratio(pts, RIGHT_EYE_IDXS)
        ear = 0.5 * (l_ear + r_ear)

        xs = (pts[:,0] * w).astype(int); ys = (pts[:,1] * h).astype(int)
        x1, y1 = max(0, xs.min()), max(0, ys.min())
        x2, y2 = min(w-1, xs.max()), min(h-1, ys.max())

        vis = bgr.copy()
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
            cx, cy = int(pts[idx,0]*w), int(pts[idx,1]*h)
            cv2.circle(vis, (cx,cy), 1, (0,255,0), -1)
        cv2.putText(vis, f"EAR:{ear:.3f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if ear < ear_thresh:
            cv2.putText(vis, "DROWSY!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        return ear, vis
