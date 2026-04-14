import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
cap = cv2.VideoCapture("alf_2.mp4")

# We want to find the keypoint that has HIGH variance during 236-271 and 279-286,
# and LOW variance in 200-230 and 290-300.
kpt_history = {i: [] for i in range(24)}
frame_types = []  # True for licking, False for background

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    if 200 <= frame_idx <= 300:
        is_licking = (236 <= frame_idx <= 271) or (279 <= frame_idx <= 286)
        frame_types.append(is_licking)
        
        results = model.predict(frame, verbose=False, max_det=1, half=True)
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Record relative Y coordinate (relative to Neck, which is 7)
            neck_y = kpts[7][1]
            for i in range(24):
                py = kpts[i][1]
                if py > 0 and neck_y > 0:
                    rel_y = py - neck_y
                    kpt_history[i].append(rel_y)
                else:
                    kpt_history[i].append(float('nan'))
        else:
            for i in range(24):
                kpt_history[i].append(float('nan'))
                
    elif frame_idx > 300:
        break
        
    frame_idx += 1

# Analyze
print("Finding the Jaw Keypoint (Index with highest correlation to Lick frames)...")
print("-" * 50)
print(f"{'Index':<6} | {'Var (Licking)':<15} | {'Var (Background)':<15} | {'Ratio':<10}")

best_kpt = -1
best_ratio = 0.0

for i in range(24):
    y_vals = np.array(kpt_history[i])
    valid_mask = ~np.isnan(y_vals)
    lick_mask = np.array(frame_types)
    
    lick_vals = y_vals[valid_mask & lick_mask]
    bg_vals = y_vals[valid_mask & ~lick_mask]
    
    if len(lick_vals) > 5 and len(bg_vals) > 5:
        var_lick = np.var(lick_vals)
        var_bg = np.var(bg_vals)
        
        ratio = var_lick / var_bg if var_bg > 0 else 0
        print(f"{i:<6} | {var_lick:<15.1f} | {var_bg:<15.1f} | {ratio:<10.2f}")
        
        if ratio > best_ratio and i != 7: # exclude neck
            best_ratio = ratio
            best_kpt = i

print("-" * 50)
print(f"Mathematical Jaw Candidate: Index {best_kpt} (Ratio: {best_ratio:.2f})")
