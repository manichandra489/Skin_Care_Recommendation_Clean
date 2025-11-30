# import cv2
# import numpy as np
# def analyze_face_contrast_bgr(frame_bgr):
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )

#     # 1) Allow larger faces by lowering minSize, and use smaller scaleFactor
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.05,      # was 1.1
#         minNeighbors=5,
#         minSize=(30, 30),      # was (100, 100)
#         flags=cv2.CASCADE_SCALE_IMAGE,
#     )

#     contrast = 0.0
#     status = "No face"

#     if len(faces) == 0:
#         return contrast, status

#     # 2) Pick largest face
#     faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
#     x, y, w, h = faces[0]
#     face_roi = gray[y:y + h, x:x + w]
#     contrast = float(np.std(face_roi))

#     if contrast < 30:
#         status = "Low"
#     elif contrast < 60:
#         status = "Medium"
#     else:
#         status = "High"

#     return contrast, status
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def analyze_face_contrast_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if not detections:
        return 0.0, "No face"

    # largest face
    faces = sorted(detections, key=lambda d: d["box"][2] * d["box"][3], reverse=True)
    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_roi = gray[y:y + h, x:x + w]

    lap = cv2.Laplacian(face_roi, cv2.CV_64F)
    focus_score = lap.var()
    dynamic_range = float(face_roi.max() - face_roi.min())

    contrast = float(
        0.7 * min(focus_score / 100.0, 1.0) +
        0.3 * min(dynamic_range / 128.0, 1.0)
    ) * 100.0

    if focus_score < 15 or dynamic_range < 20:
        status = "Low"
    elif contrast < 40:
        status = "Medium"
    else:
        status = "High"

    return contrast, status





