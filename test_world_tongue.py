import cv2
import os
import numpy as np
from ultralytics import YOLO, YOLOWorld

print("Initializing Research Pipeline...")
pose_model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
world_model = YOLOWorld("yolov8s-world.pt").to("mps")
world_model.set_classes(["dog tongue"])

cap = cv2.VideoCapture("alf_2.mp4")
frame_idx = 0

os.makedirs("research_crops", exist_ok=True)
success_count = 0

print("Testing Licking frames 236-286...")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # We test both licking sequences: 236-271 and 279-286
    if (236 <= frame_idx <= 271) or (279 <= frame_idx <= 286):
        # 1. Get Nose Point
        results_pose = pose_model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results_pose) > 0 and results_pose[0].keypoints is not None:
            try:
                nose_x = int(results_pose[0].keypoints.xy[0][0][0].item())
                nose_y = int(results_pose[0].keypoints.xy[0][0][1].item())
                
                # 2. Crop 160x160 region (Mouth is typically below/around Nose)
                # We shift it slightly down because the mouth is beneath the nose
                crop_size = 80
                y1 = max(0, nose_y - 20)
                y2 = min(frame.shape[0], nose_y + 140)
                x1 = max(0, nose_x - crop_size)
                x2 = min(frame.shape[1], nose_x + crop_size)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # 3. Predict Tongue
                    results_world = world_model.predict(crop, verbose=False, conf=0.05)
                    
                    detected = False
                    if len(results_world) > 0 and len(results_world[0].boxes) > 0:
                        detected = True
                        success_count += 1
                        
                        # Draw for visual inspection if needed
                        for box in results_world[0].boxes:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(crop, (bx1, by1), (bx2, by2), (255, 0, 255), 2)
                    
                    # Save crop for review
                    cv2.imwrite(f"research_crops/frame_{frame_idx:03d}_{'DET' if detected else 'miss'}.jpg", crop)
                    print(f"Frame {frame_idx:03d}: Tongue Detected = {detected}")
            except Exception as e:
                print(f"Error frame {frame_idx}: {e}")
    elif frame_idx > 286:
        break
        
    frame_idx += 1

print(f"Total detections: {success_count}")
