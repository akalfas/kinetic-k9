import cv2
import time
from anxiety_engine import YOLO26AnxietyEngine

def test():
    # Initialize the engine
    try:
        # Based on the user's files we see yolo26m-pose.pt or runs/pose/training_run_1/weights/best.pt
        import os
        model_path = "runs/pose/training_run_1/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = "yolo26m-pose.pt"
        
        print(f"Loading engine with model: {model_path}")
        engine = YOLO26AnxietyEngine(model_path=model_path)
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    # Open video
    video_path = "alf_1.mov"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    print("Running test over 50 frames...")
    
    frame_count = 0
    total_time = 0.0
    
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_t = time.time()
        score, state, metrics = engine.process_frame(frame)
        end_t = time.time()
        
        frame_time = (end_t - start_t) * 1000
        total_time += frame_time
        
        print(f"Frame {frame_count:02d} | Time: {frame_time:.1f}ms | State: {state} | Score: {score:.2f} | Metrics: {metrics}")
        
        frame_count += 1

    print(f"\nAverage Frame Time: {total_time / frame_count:.1f}ms")
    cap.release()

if __name__ == "__main__":
    test()
