import cv2
import sys
import numpy as np
from anxiety_engine import YOLO26AnxietyEngine

engine = YOLO26AnxietyEngine()
cap = cv2.VideoCapture("alf_2.mp4")

frame_idx = 0
found_triggers = []

print("Running pure engine verification on alf_2.mp4...")
print("-" * 40)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    score, state, metrics = engine.process_frame(frame)
    is_licking = metrics.get('licking', False) if metrics else False
    
    if is_licking:
        found_triggers.append(frame_idx)
        print(f"LICK DETECTED @ Frame {frame_idx:03d} | Score: {score:.2f} | Amplitude: {metrics.get('lick_amplitude', 0):.1f} | Cycles: {metrics.get('lick_cycles', 0)}")
        
    frame_idx += 1

print("-" * 40)
print(f"Total licking frames flagged: {len(found_triggers)}")
print(f"Triggered Frames List: {found_triggers}")
