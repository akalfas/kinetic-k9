import cv2
from ultralytics import YOLO

det = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("test_pointer.f137.mp4")

found_human = False
found_dog = False

for i in range(1000):
   ret, frame = cap.read()
   if not ret: break
   res = det.predict(frame, verbose=False)
   
   for box in res[0].boxes:
       cls_id = int(box.cls[0].item())
       if cls_id == 0 and not found_human:
           print(f"Frame {i}: Found Human (0)", flush=True)
           found_human = True
       if cls_id == 16 and not found_dog:
           print(f"Frame {i}: Found Dog (16)", flush=True)
           found_dog = True
       if found_human and found_dog:
           break
   if found_human and found_dog:
       print("Found both in first 1000 frames!", flush=True)
       break
print("Done.", flush=True)
