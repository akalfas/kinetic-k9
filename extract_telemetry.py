import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load model
model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")

cap = cv2.VideoCapture("alf_2.mp4")
frame_idx = 0

window = deque(maxlen=30)

print("Frame | Rel_Y | dy_sum_abs | cycles")
print("-" * 40)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    if 200 <= frame_idx <= 300:
        results = model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Index 0 is Nose, 7 is Neck. (Assuming Jaw was not provided)
            # Actually, I should print Nose[1] and Neck[1] to understand scaling.
            try:
                nose_y = kpts[0][1]
                neck_y = kpts[7][1]
                if nose_y > 0 and neck_y > 0:
                    rel_y = nose_y - neck_y
                    window.append(rel_y)
                    
                    cycles = 0
                    dy_mag = 0
                    if len(window) == 30:
                        y_arr = np.array(window)
                        dy = np.diff(y_arr)
                        sign_changes = np.where(np.diff(np.signbit(dy)))[0]
                        cycles = len(sign_changes) / 2.0
                        dy_mag = np.sum(np.abs(dy))
                        
                    tag = ""
                    if 236 <= frame_idx <= 271:
                        tag = "<-- LICK 1"
                    elif 279 <= frame_idx <= 286:
                        tag = "<-- LICK 2 (Fast)"
                        
                    print(f"{frame_idx:03d} | {rel_y:6.1f} | {dy_mag:6.1f} | {cycles:4.1f} {tag}")
            except Exception as e:
                pass
                
    frame_idx += 1
    if frame_idx > 300:
        break
