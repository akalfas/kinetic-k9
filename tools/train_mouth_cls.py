from ultralytics import YOLO

def main():
    print("Loading YOLO11 Nano Classifier Architecture...")
    model = YOLO('yolo11n-cls.pt')

    print("Beginning Training on Apple Silicon (MPS)...")
    model.train(
        data='datasets/mouth_cls',
        epochs=40,
        imgsz=160,
        device='mps',
        project='runs/cls',
        name='mouth_run',
        patience=10
    )
    print("Training Completed Successfully!")

if __name__ == '__main__':
    main()
