import cv2
from ultralytics import YOLO

det_n = YOLO("yolo11n.pt")
det_m = YOLO("yolo11m.pt")
cap = cv2.VideoCapture("test_pointer.f137.mp4")

# Skip to frame 300
for _ in range(300):
   ret, frame = cap.read()

res_n = det_n.predict(frame, verbose=False, classes=[16], conf=0.1)
res_m = det_m.predict(frame, verbose=False, classes=[16], conf=0.1)

print("Nano Model (conf=0.1):", res_n[0].boxes.xywh.cpu().numpy())
print("Medium Model (conf=0.1):", res_m[0].boxes.xywh.cpu().numpy())
