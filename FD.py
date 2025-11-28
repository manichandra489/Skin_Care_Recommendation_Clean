import cv2
import numpy as np
import mediapipe as mp
import time
import os

mp_face = mp.solutions.face_detection

def compute_contrast(roi_bgr, use_clahe=True):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    return float(np.std(gray))

def color_for_contrast(c, low=25.0, high=60.0):
    # Tune low/high or make dynamic
    if c < low:
        return (0, 0, 255), "Low"      # Red (BGR)
    elif c < high:
        return (0, 255, 255), "Medium" # Yellow
    else:
        return (0, 255, 0), "High"     # Green

def smooth_box(prev, curr, alpha=0.6):
    if prev is None:
        return curr
    x = int(alpha * prev[0] + (1-alpha) * curr[0])
    y = int(alpha * prev[1] + (1-alpha) * curr[1])
    w = int(alpha * prev[2] + (1-alpha) * curr[2])
    h = int(alpha * prev[3] + (1-alpha) * curr[3])
    return (x, y, w, h)

def capture_face_contrast_mediapipe(save_path="face_contrast.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prev_boxes = []  # store previous boxes to smooth per-index (simple)
    smoothing_alpha = 0.6
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    print("Press SPACE to capture, ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        boxes = []
        scores = []

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # expand a little to include hair/forehead if desired
                pad_x = int(0.1 * bw)
                pad_y = int(0.15 * bh)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                boxes.append((x1, y1, x2-x1, y2-y1))
                scores.append(float(det.score[0]) if det.score else 0.0)

        # simple smoothing: match by index (MediaPipe usually keeps order stable)
        smoothed = []
        for i, box in enumerate(boxes):
            prev = prev_boxes[i] if i < len(prev_boxes) else None
            sm = smooth_box(prev, box, alpha=smoothing_alpha)
            smoothed.append(sm)
        prev_boxes = smoothed
        contrast = 0

        # Draw boxes and compute contrast per face
        for (x, y, bw, bh) in smoothed:
            # Ensure ROI inside image
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            roi = frame[y:y2, x:x2].copy()
            if roi.size == 0:
                continue
            contrast = compute_contrast(roi, use_clahe=True)
            color, status = color_for_contrast(contrast, low=25.0, high=60.0)

            # Draw rectangular outline
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            # Draw filled indicator bar above the box
            bar_h = 28
            cv2.rectangle(frame, (x, max(0,y-bar_h)), (x+160, y), color, -1)
            label = f"{status} {contrast:.1f}"
            cv2.putText(frame, label, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Face Contrast (MediaPipe)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if contrast > 60:
                return frame
            print("Contrast too low, try again.")
            continue        
            # save current frame
            # abs_path = os.path.abspath(save_path)
            # cv2.imwrite(abs_path, frame)
            # print("Saved:", abs_path)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()




