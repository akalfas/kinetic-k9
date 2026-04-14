import cv2
import os
import random
import shutil
import numpy as np
from ultralytics import YOLO

def setup_dirs(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    for split in ['train', 'val']:
        for cls in ['licking', 'closed']:
            os.makedirs(os.path.join(base_path, split, cls), exist_ok=True)

def get_crop(frame, kpts):
    nose_x, nose_y = kpts[16][0], kpts[16][1] # Official index for Nose
    if nose_x == 0 or nose_y == 0:
        return None
    
    # 160x160 crop shifted downward to capture the jaw/tongue
    y1 = int(max(0, nose_y - 40))
    y2 = int(min(frame.shape[0], nose_y + 120))
    x1 = int(max(0, nose_x - 80))
    x2 = int(min(frame.shape[1], nose_x + 80))
    
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0 and crop.shape[0] > 50 and crop.shape[1] > 50:
        return crop
    return None

def main():
    model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
    base_path = "datasets/mouth_cls"
    setup_dirs(base_path)
    
    licking_crops = []
    closed_crops = []
    
    # Process alf_2.mp4 (Contains both licking and closed)
    print("Processing alf_2.mp4...")
    cap = cv2.VideoCapture("alf_2.mp4")
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Determine strict bounds
        is_licking = (236 <= f_idx <= 271) or (279 <= f_idx <= 286)
        is_closed = (20 <= f_idx <= 230) or (290 <= f_idx <= 305) # Safe stable frames
        
        if is_licking or is_closed:
            res = model.predict(frame, verbose=False, max_det=1, half=True)
            if len(res) > 0 and res[0].keypoints is not None and len(res[0].keypoints.xy) > 0:
                crop = get_crop(frame, res[0].keypoints.xy[0].cpu().numpy())
                if crop is not None:
                    if is_licking:
                        licking_crops.append(crop)
                    else:
                        closed_crops.append(crop)
        f_idx += 1
    cap.release()

    # Process alf_1.mov for negative context (All closed)
    print("Processing alf_1.mov...")
    cap = cv2.VideoCapture("alf_1.mov")
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Only take around 60 frames evenly to balance dataset
        if f_idx % 2 == 0 and f_idx < 120:
            res = model.predict(frame, verbose=False, max_det=1, half=True)
            if len(res) > 0 and res[0].keypoints is not None and len(res[0].keypoints.xy) > 0:
                crop = get_crop(frame, res[0].keypoints.xy[0].cpu().numpy())
                if crop is not None:
                    closed_crops.append(crop)
        f_idx += 1
    cap.release()
    
    print(f"Extracted {len(licking_crops)} licking, {len(closed_crops)} closed.")
    
    # Optional balancing if vastly outnumbered
    target_negative = int(len(licking_crops) * 2.5) # Allow 2.5x negative to bias against false positives
    if len(closed_crops) > target_negative:
        random.shuffle(closed_crops)
        closed_crops = closed_crops[:target_negative]
        
    print(f"Balanced Dataset: {len(licking_crops)} licking, {len(closed_crops)} closed.")

    # Save to disk
    def save_crops(crops, label_name):
        random.shuffle(crops)
        split_idx = int(len(crops) * 0.8)
        
        for i, crop in enumerate(crops):
            split = 'train' if i < split_idx else 'val'
            path = os.path.join(base_path, split, label_name, f"{label_name}_{i:04d}.jpg")
            # We resize natively so the directory holds uniform 160x160 images
            rsz = cv2.resize(crop, (160, 160))
            cv2.imwrite(path, rsz)

    save_crops(licking_crops, "licking")
    save_crops(closed_crops, "closed")
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
