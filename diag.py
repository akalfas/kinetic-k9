import cv2, numpy as np
from collections import deque
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt")
cap = cv2.VideoCapture('alf_2.mp4')
velocity_history = deque(maxlen=20)
area_history = deque(maxlen=20)

frame_count = 0
prev_cx, prev_cy = None, None
current_state = "ANALYZING..."
previous_state = "ANALYZING..."

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes.xywh.cpu().numpy()
    
    if len(boxes) > 0:
        cx, cy, w, h = boxes[0]
        area = w * h
        area_history.append(area)
        
        dist = 0
        if prev_cx is not None and prev_cy is not None:
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            velocity_history.append(dist)
        
        prev_cx, prev_cy = cx, cy

        if len(velocity_history) == 20:
            avg_speed = np.mean(velocity_history)
            shape_shift = np.std(area_history) / np.mean(area_history) * 100
            
            if avg_speed < 5.0:
                activity_score = avg_speed + (shape_shift * 0.1)
            else:
                activity_score = avg_speed + shape_shift
            
            if previous_state == "STATE: ACTIVE":
                threshold = 15 
            else:
                threshold = 28 
            
            if activity_score > threshold:
                current_state = "STATE: ACTIVE"
            else:
                current_state = "STATE: CALM"
                
            if current_state != previous_state and previous_state != "ANALYZING...":
                 print(f"==> FLIP TO {current_state} at frame {frame_count}")
                 
            if dist > 15: # Significant movement
                 print(f"Frame {frame_count}: Moved {dist:.1f}px. AvgSpd: {avg_speed:.1f}, Shape: {shape_shift:.1f}, Score: {activity_score:.1f}, {current_state}")
                 
            previous_state = current_state

    frame_count += 1

