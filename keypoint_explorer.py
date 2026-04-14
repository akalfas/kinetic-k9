import cv2
import sys
import os
import time
from ultralytics import YOLO

def explore_keypoints(video_path):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    # Load Model
    model_path = "runs/pose/training_run_1/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo26m-pose.pt"
    print(f"Loading Model: {model_path}...")
    model = YOLO(model_path).to("mps")

    print(f"Buffering '{video_path}' into memory for lag-free scrubbing...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("Video is empty.")
        return
        
    print(f"Buffered {len(frames)} frames successfully.")
    
    paused = True
    delay = 100
    frame_idx = 0
    total_frames = len(frames)
    
    print("\n--- Kinetic K9 Keypoint Explorer ---")
    print("Controls:")
    print("  [SPACE] - Pause / Play")
    print("  [ d ]   - Step 1 frame FORWARD (when paused)")
    print("  [ a ]   - Step 1 frame BACKWARD (when paused)")
    print("  [ q ]   - Quit")
    
    # Needs to process explicitly if paused
    force_update = True

    while True:
        if force_update or not paused:
            raw_frame = frames[frame_idx]
            display_frame = raw_frame.copy()
            
            # Predict
            results = model.predict(raw_frame, verbose=False, max_det=1, half=True)
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                for i, (px, py) in enumerate(kpts):
                    if px > 0 and py > 0:
                        cv2.circle(display_frame, (int(px), int(py)), 3, (0, 165, 255), -1)
                        cv2.putText(display_frame, str(i), (int(px)+5, int(py)-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(display_frame, str(i), (int(px)+5, int(py)-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    
            cv2.putText(display_frame, f"FRAME: {frame_idx} / {total_frames - 1}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Add scrubbing instructions if paused
            if paused:
                cv2.putText(display_frame, "PAUSED - USE 'a' and 'd' TO SCRUB", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
            cv2.imshow("Keypoint Sandbox", display_frame)
            force_update = False
            
            if not paused:
                if frame_idx < total_frames - 1:
                    frame_idx += 1
                else:
                    paused = True
                    force_update = True

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            force_update = True
        elif key == ord('d') and paused:
            frame_idx = min(total_frames - 1, frame_idx + 1)
            force_update = True
        elif key == ord('a') and paused:
            frame_idx = max(0, frame_idx - 1)
            force_update = True

    cv2.destroyAllWindows()

if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else "alf_1.mov"
    explore_keypoints(target_video)
