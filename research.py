import cv2
import os
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt")
cap = cv2.VideoCapture("alf_1.mov")

# Skip to frame 30 where the dog is clearly lying down, giving us a good view of the points
for _ in range(30):
    cap.read()
    
ret, frame = cap.read()
if ret:
    results = model.predict(frame, verbose=False)
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    
    # Draw index numbers over the skeleton
    for i, (x, y) in enumerate(kpts):
        if x > 0 and y > 0: # Ensure point is detected
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)
            cv2.putText(frame, str(i), (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
    cv2.imwrite("keypoint_map.jpg", frame)
    print("Found keypoints length:", len(kpts))
else:
    print("Failed to read frame")
