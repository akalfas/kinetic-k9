import cv2, numpy as np
from collections import deque
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt")
cap = cv2.VideoCapture("test_pointer.f137.mp4")
keypoint_history = deque(maxlen=20)

for frame_idx in range(600):
    ret, frame = cap.read()
    if not ret: break
    # Skip early frames to get to the action
    if frame_idx < 100: continue
    
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes.xywh.cpu().numpy()
    if len(boxes) > 0:
        valid = [i for i, b in enumerate(boxes) if b[2] >= b[3] * 0.8]
        if not valid: valid = range(len(boxes))
        areas = [boxes[i,2] * boxes[i,3] for i in valid]
        best_idx = valid[np.argmax(areas)]
        
        cx, cy, w, h = boxes[best_idx]
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > best_idx:
            raw_kpts = results[0].keypoints.xy[best_idx].cpu().numpy()
            kpts_relative = []
            for px, py in raw_kpts:
                if px > 0 and py > 0:
                    kpts_relative.extend([px - cx, py - cy])
                else:
                    kpts_relative.extend([0.0, 0.0])
            keypoint_history.append(kpts_relative)
            
            if len(keypoint_history) == 20 and frame_idx % 30 == 0:
                history_stack = np.array(keypoint_history)
                total_jitter = np.sum(np.var(history_stack, axis=0))
                print(f"Frame {frame_idx} | Aspect: {w/h:.2f} | Jitter: {total_jitter:.1f}")
