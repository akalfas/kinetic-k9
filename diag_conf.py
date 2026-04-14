import cv2, numpy as np
from ultralytics import YOLO

pose = YOLO("runs/pose/training_run_1/weights/best.pt")
cap = cv2.VideoCapture("test_pointer.f137.mp4")

for _ in range(500): cap.read()
ret, frame = cap.read()
res = pose.predict(frame, verbose=False)
boxes = res[0].boxes.xywh.cpu().numpy()
conf = res[0].boxes.conf.cpu().numpy()

for i, (cx, cy, w, h) in enumerate(boxes):
    print(f"Box {i} | Conf: {conf[i]:.2f} | w/h: {w/h:.2f} | Area: {w*h:.0f}", flush=True)

