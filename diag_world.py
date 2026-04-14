import cv2
from ultralytics import YOLO

model = YOLO("yolov8s-world.pt")
model.set_classes(["pointing dog", "hunter", "grass"])

cap = cv2.VideoCapture("test_pointer.f137.mp4")
for _ in range(300):
   ret, frame = cap.read()

res = model.predict(frame, verbose=False)
for box in res[0].boxes:
    cls_id = int(box.cls[0].item())
    name = model.names[cls_id]
    conf = box.conf[0].item()
    print(f"Found: {name} | Conf: {conf:.2f} | Box: {box.xywh.cpu().numpy()}", flush=True)

