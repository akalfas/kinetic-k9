import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
cap = cv2.VideoCapture("alf_2.mp4")

frame_idx = 0
window = deque(maxlen=30)
found_triggers = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model.predict(frame, verbose=False, max_det=1, half=True)
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        nose_y, neck_y = kpts[0][1], kpts[7][1]
        
        if nose_y > 0 and neck_y > 0:
            window.append(nose_y - neck_y)
            
            if len(window) == 30:
                y_arr = np.array(window)
                
                # Apply 5-frame moving average low-pass filter
                smooth_y = np.convolve(y_arr, np.ones(5)/5.0, mode='valid')
                
                # Now calculate derivatives on smoothed data
                dy = np.diff(smooth_y)
                sign_changes = np.where(np.diff(np.signbit(dy)))[0]
                cycles = len(sign_changes) / 2.0
                
                # Check amplitude of the smoothed signal
                smooth_range = np.max(smooth_y) - np.min(smooth_y)
                
                # Are we seeing dense rhythmic cycles in the smooth signal?
                if cycles >= 1.5 and smooth_range > 15.0:
                    found_triggers.append(frame_idx)
                    
    frame_idx += 1

print(f"Total Triggers: {len(found_triggers)}")
print("Triggered on Frames:", found_triggers)

