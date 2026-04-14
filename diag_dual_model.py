import cv2
from ultralytics import YOLO

# Load both models
detector = YOLO("yolo11n.pt")
pose = YOLO("runs/pose/training_run_1/weights/best.pt")

cap = cv2.VideoCapture("test_pointer.f137.mp4")

for _ in range(120):
    ret, frame = cap.read()

# Run detector
det_results = detector.predict(frame, verbose=False)
boxes = det_results[0].boxes
for box in boxes:
    cls_id = int(box.cls[0].item())
    name = detector.names[cls_id]
    print(f"Detector found: {name} at {box.xyxy[0].tolist()}")

# Run pose
pose_results = pose.predict(frame, verbose=False)
p_boxes = pose_results[0].boxes
for p_box in p_boxes:
    print(f"Pose Model found animal at: {p_box.xyxy[0].tolist()}")

