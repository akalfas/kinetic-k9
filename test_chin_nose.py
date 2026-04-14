import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
cap = cv2.VideoCapture("alf_2.mp4")

frame_idx = 0
prev_dist = 0

print(f"{'Frame':<8} | {'Nose_Y':<8} | {'Chin_Y':<8} | {'Dist':<8} | {'dy':<8} | Status")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    if 200 <= frame_idx <= 300:
        results = model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Using the official dog-pose YAML order
            nose_y = kpts[16][1]
            chin_y = kpts[17][1]
            
            if nose_y > 0 and chin_y > 0:
                dist = chin_y - nose_y # Chin is ideally below the nose physically
                dy = dist - prev_dist
                # We specifically care about the Lick Frames
                status = "<-- LICK" if (236 <= frame_idx <= 271) or (279 <= frame_idx <= 286) else ""
                
                print(f"{frame_idx:<8d} | {nose_y:<8.1f} | {chin_y:<8.1f} | {dist:<8.1f} | {dy:<8.1f} | {status}")
                prev_dist = dist
                
    elif frame_idx > 300:
        break
        
    frame_idx += 1
