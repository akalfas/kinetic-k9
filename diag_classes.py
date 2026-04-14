from ultralytics import YOLO
model = YOLO("runs/pose/training_run_1/weights/best.pt")
print("Model classes:", model.names)
