import cv2
import numpy as np

# def analyze_face_contrast_bgr(frame_bgr):
#     """Input: BGR image (numpy array). Returns: (annotated_bgr, contrast, status)."""
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
#     )


#     contrast = 0
#     status = "No face"
#     annotated = frame_bgr.copy()

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         contrast = float(np.std(face_roi))

#         if contrast < 30:
#             color, status = (0, 0, 255), "Low"
#         elif contrast < 60:
#             color, status = (0, 255, 255), "Medium"
#         else:
#             color, status = (0, 255, 0), "High"

#         cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)
#         cv2.putText(
#             annotated,
#             f"Contrast: {contrast:.1f} ({status})",
#             (x, y-10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             color,
#             2,
#         )

#     return  contrast, status
def analyze_face_contrast_bgr(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 1) Allow larger faces by lowering minSize, and use smaller scaleFactor
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,      # was 1.1
        minNeighbors=5,
        minSize=(30, 30),      # was (100, 100)
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    contrast = 0.0
    status = "No face"

    if len(faces) == 0:
        return contrast, status

    # 2) Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    contrast = float(np.std(face_roi))

    if contrast < 30:
        status = "Low"
    elif contrast < 60:
        status = "Medium"
    else:
        status = "High"

    return contrast, status



