"""
Collect REAL training samples from your webcam.

Shows each gesture label one at a time; hold the sign in front of the camera
and the script records the MediaPipe hand landmarks into data/landmarks.csv.
Then retrain with:  python model/train_model.py

Needs the optional deps:  pip install opencv-python mediapipe
Usage:  python model/collect_data.py [samples_per_gesture]
Keys:   SPACE = start/pause recording for the current gesture, Q = quit
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gestures import LABELS, normalize  # noqa: E402

try:
    import cv2
    import mediapipe as mp
except ImportError:
    sys.exit("This script needs: pip install opencv-python mediapipe")

PER_GESTURE = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "landmarks.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
rows = []
print("Gestures:", ", ".join(LABELS))
for label in LABELS:
    print(f"\nShow the sign for '{label}'. Press SPACE to record {PER_GESTURE} samples, Q to quit.")
    recording, count = False, 0
    while count < PER_GESTURE:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lm = [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
            if recording:
                rows.append([label] + [f"{v:.4f}" for v in normalize(lm)])
                count += 1
                time.sleep(0.05)
        cv2.putText(frame, f"{label}: {count}/{PER_GESTURE}  SPACE=record  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SignSpeak - collect data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif key == ord("q"):
            recording = None
            break
    if recording is None:
        break

with open(OUT, "a", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"Saved {len(rows)} samples to {OUT}")
cap.release()
cv2.destroyAllWindows()
