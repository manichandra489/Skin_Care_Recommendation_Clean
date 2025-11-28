import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import FD


def capture_face_contrast():

    # ONE invisible root, never deiconify it
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        # just call the dialog; no lift/deiconify
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
            parent=root
        )

        if not file_path:
            print("❌ No file selected. Exiting image selection.")
            break

        frame = cv2.imread(file_path)
        if frame is None:
            print("❌ Could not read image. Try another.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        contrast = 0

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            contrast = np.std(face_roi)

            if contrast < 30:
                color, status = (0, 0, 255), "Low"
            elif contrast < 60:
                color, status = (0, 255, 255), "Medium"
            else:
                color, status = (0, 255, 0), "High"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                frame,
                f"Contrast: {contrast:.1f} ({status})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", frame)
        print("Press SPACE to select another file, ESC to exit.")

        while True:
            k = cv2.waitKey(50) & 0xFF
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                if contrast > 60:
                    root.destroy()
                    return frame
                k = 27  # treat close as ESC
                break
            if k in (27, 32):
                break

        cv2.destroyAllWindows()

        if k == 27:
            print("Exiting (ESC or window closed).")
            break
        elif k == 32:
            if contrast > 60:
                root.destroy()
                return frame
            print("Select another image (SPACE pressed).")
            continue

    root.destroy()

       
capture_face_contrast()



