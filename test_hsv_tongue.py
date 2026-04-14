import cv2
import numpy as np
from ultralytics import YOLO

pose_model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
cap = cv2.VideoCapture("alf_2.mp4")
frame_idx = 0

print("Frame | Pixels | Status")
print("-" * 30)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    if 200 <= frame_idx <= 300:
        results = pose_model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            try:
                nose_x = int(kpts[0][0])
                nose_y = int(kpts[0][1])
                
                # Crop area precisely below the nose! 
                # The tongue extends downward and sometimes slightly up, but mostly below the nose line
                crop_size_x = 60
                y1 = nose_y
                y2 = min(frame.shape[0], nose_y + 100)
                x1 = max(0, nose_x - crop_size_x)
                x2 = min(frame.shape[1], nose_x + crop_size_x)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    
                    # Define pink/red color boundary in HSV
                    # Red wraps around the HSV spectrum (0-10 and 160-180)
                    lower_red1 = np.array([0, 50, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([160, 50, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    pink_mask = mask1 | mask2
                    
                    pink_pixels = cv2.countNonZero(pink_mask)
                    
                    is_lick = (236 <= frame_idx <= 271) or (279 <= frame_idx <= 286)
                    status = "<-- LICK" if is_lick else ""
                    
                    print(f"{frame_idx:03d} | {pink_pixels:6d} | {status}")
            except Exception as e:
                pass
                
    elif frame_idx > 300:
        break
        
    frame_idx += 1
