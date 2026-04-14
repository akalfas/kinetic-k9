from ultralytics import YOLO

def main():
    print("Initializing YOLO26 pose model...")
    # Load the base pose model
    model = YOLO("yolo26m-pose.pt")
    
    print("Starting training on the dog-pose dataset...")
    # Train the model. 
    # Notice: It will automatically download dog-pose.yaml and the required images.
    # We use 50 epochs for the PoC to speed things up, but you can increase it later.
    model.train(
        data="dog-pose.yaml",
        epochs=50,
        imgsz=640,
        device="mps"  # Uses your Mac's Metal Performance Shaders (GPU)
    )
    print("Training finished! Find the best model at 'runs/pose/train/weights/best.pt'")

if __name__ == "__main__":
    main()
