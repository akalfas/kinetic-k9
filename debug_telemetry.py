import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
cap = cv2.VideoCapture("alf_2.mp4")

frame_idx = 0
prev_rel_y = None

print("Frame | rel_y | dy")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    if 230 <= frame_idx <= 290:
        results = model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            try:
                nose_y = kpts[0][1]
                neck_y = kpts[7][1]
                if nose_y > 0 and neck_y > 0:
                    rel_y = nose_y - neck_y
                    dy = rel_y - prev_rel_y if prev_rel_y is not None else 0
                    print(f"{frame_idx:3d} | {rel_y:6.1f} | {dy:6.1f}")
                    prev_rel_y = rel_y
            except:
                pass
    elif frame_idx > 290:
        break
    else:
        # Keep tracking prev_rel_y even outside print range
        results = model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            try:
                nose_y = kpts[0][1]
                neck_y = kpts[7][1]
                if nose_y > 0 and neck_y > 0:
                    prev_rel_y = nose_y - neck_y
            except:
                pass
                
    frame_idx += 1
