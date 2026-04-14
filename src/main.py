import cv2
import sys
import os
import csv
import numpy as np
from collections import deque
from ultralytics import YOLO

def play_video(video_path):
    print(f"Attempting to open video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Please place a video file here or update the path in the script.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Load dynamic open-vocabulary gatekeeper model
    print("Loading YOLO-World dynamic gatekeeper...")
    try:
        detector_model = YOLO("yolov8s-world.pt").to("mps")
    except Exception:
        detector_model = YOLO("yolov8s-world.pt")
    detector_model.set_classes(["pointer dog", "hunting dog"])

    # Load specialized pose model
    print("Loading custom trained dog-pose model...")
    try:
        pose_model = YOLO("runs/pose/training_run_1/weights/best.pt").to("mps")
    except Exception:
        pose_model = YOLO("runs/pose/training_run_1/weights/best.pt")

    # Get frame rate for the video writers
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # fallback
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup Video Writers for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_normal_path = f"{base_name}_output_normal.mp4"
    out_slow_path = f"{base_name}_output_slow.mp4"
    
    out_normal = cv2.VideoWriter(out_normal_path, fourcc, fps, (width, height))
    # For slow motion, we write exactly the same frames, but tell the video file to play them back at 30% speed
    out_slow = cv2.VideoWriter(out_slow_path, fourcc, fps * 0.3, (width, height))
    
    print(f"Rendering outputs to {out_normal_path} and {out_slow_path}...")

    # Sliding window to track the movement and shape of the dog
    # y_history was too simple! We need to track 2D movement (Velocity) and Shape (Area)
    velocity_history = deque(maxlen=20)
    area_history = deque(maxlen=20)
    
    # NEW: Track raw skeletal coordinates to measure physical rigidity
    keypoint_history = deque(maxlen=20)
    
    # Create directory to isolate transition images
    output_dir = "saved_transitions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Open CSV file to log telemetry data
    csv_file = open('telemetry.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Write headers
    csv_writer.writerow(['frame', 'activity_score', 'state'])
    
    current_state = "ANALYZING..."
    previous_state = "ANALYZING..."
    prev_cx, prev_cy = None, None
    frame_count = 0
    
    # NEW: Cache the heavy bounding box model to save 90% compute
    cached_gatekeeper_box = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video playback finished.")
            break

        # --- Predictive Caching Gatekeeper Pass ---
        # The World model uses massive transformers. Only run it every 10 frames (or if lost).
        if cached_gatekeeper_box is None or frame_count % 10 == 0:
            det_results = detector_model.predict(frame, verbose=False, conf=0.1)
            det_boxes = det_results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            
            if len(det_boxes) > 0:
                best_det_idx = np.argmax(det_results[0].boxes.conf.cpu().numpy())
                bx1, by1, bx2, by2 = det_boxes[best_det_idx]
                
                # Expand box by 15% to create a generous tracking zone padding
                pad_x = (bx2 - bx1) * 0.15
                pad_y = (by2 - by1) * 0.15
                cached_gatekeeper_box = [bx1 - pad_x, by1 - pad_y, bx2 + pad_x, by2 + pad_y]
            else:
                cached_gatekeeper_box = None
                
        # --- Specialized Pose Pass ---
        # The pose model is fast enough to run every frame
        pose_results = pose_model.predict(frame, verbose=False)
        annotated_frame = frame.copy()
        
        boxes = pose_results[0].boxes
        pose_boxes = boxes.xywh.cpu().numpy() # [cx, cy, w, h]
        
        valid_pose_idx = -1
        
        # Spatial Filter: Match Skeletons to Cached Target Zone
        if cached_gatekeeper_box is not None and len(pose_boxes) > 0:
            bx1, by1, bx2, by2 = cached_gatekeeper_box
            
            # Find the skeletal pose whose center falls inside the Target Zone
            for i, (cx, cy, w, h) in enumerate(pose_boxes):
                if bx1 < cx < bx2 and by1 < cy < by2:
                    valid_pose_idx = i
                    break # Lock on first match
            
        activity_score = 0
        total_jitter = 1000
        
        if valid_pose_idx != -1:
            best_box = boxes[valid_pose_idx].xywh[0]
            cx, cy, w, h = best_box[0].item(), best_box[1].item(), best_box[2].item(), best_box[3].item()
            area = w * h
            
            # --- Visual Rendering ---
            # Draw ONLY the verified Dog Bounding Box (Hide hallucinated humans)
            cv2.rectangle(annotated_frame, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (255, 0, 0), 2)
            
            # Extract Skeletal Keypoints for Mathematics
            if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > valid_pose_idx:
                raw_kpts = pose_results[0].keypoints.xy[valid_pose_idx].cpu().numpy()
                
                # Draw Verified Skeleton purely on the focal dog
                valid_kpts_x = [px for px, py in raw_kpts if px > 0]
                valid_kpts_y = [py for px, py in raw_kpts if py > 0]
                
                for px, py in raw_kpts:
                    if px > 0 and py > 0:
                        cv2.circle(annotated_frame, (int(px), int(py)), 5, (0, 255, 255), -1)
                        
                # Immunize against Camera Panning
                # We anchor the points to the Skeletal Center-of-Mass, NOT the volatile Bounding Box!
                kpts_relative = []
                if len(valid_kpts_x) > 0:
                    skel_cx = np.mean(valid_kpts_x)
                    skel_cy = np.mean(valid_kpts_y)
                    
                    for px, py in raw_kpts:
                        if px > 0 and py > 0:
                            kpts_relative.append(px - skel_cx)
                            kpts_relative.append(py - skel_cy)
                        else:
                            kpts_relative.append(0.0)
                            kpts_relative.append(0.0)
                            
                    keypoint_history.append(kpts_relative)
            
            # 1. Track Shape Shifting (Area variance)
            area_history.append(area)
            
            # 2. Track 2D Velocity (Distance moved from last frame)
            if prev_cx is not None and prev_cy is not None:
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                velocity_history.append(dist)
            
            prev_cx, prev_cy = cx, cy

            # Once we have enough history, calculate the Activity Score
            if len(velocity_history) == 20:
                # Average speed over the last 20 frames
                avg_speed = np.mean(velocity_history)
                
                # How much is the dog's shape twisting/changing? (Normalized variance of area)
                shape_shift = np.std(area_history) / np.mean(area_history) * 100
                
                # Combined heuristic score
                # If the dog is basically stationary (avg_speed < 5.0), 
                # heavily discount shape shifts (like chewing a ball while laying down).
                if avg_speed < 5.0:
                    activity_score = avg_speed + (shape_shift * 0.1)
                else:
                    activity_score = avg_speed + shape_shift
                
                # Apply Hysteresis to completely eliminate frame-flickering
                if previous_state == "STATE: ACTIVE":
                    # Must significantly calm down to drop the state
                    threshold = 15 
                else:
                    # Must have a clear spike to trigger active
                    threshold = 28 
                
                # Calculate internal structural jitter among the 20 specific joints
                total_jitter = 1000 # Default high
                if len(keypoint_history) == 20:
                    history_stack = np.array(keypoint_history)
                    joint_variances = np.var(history_stack, axis=0) 
                    total_jitter = np.sum(joint_variances)
                
                # FIX 2: Even if a dog is completely frozen rigid in a field, the cameraman 
                # walking around them causes the dog to "travel" across the camera lens (high avg_speed).
                # Since we immunized the skeleton by mapping to its internal Center-of-Mass, we can detect perfect freezes!
                if total_jitter < 3500.0:
                    current_state = "STATE: FIXATED"
                    color = (0, 165, 255) # Orange Text
                elif activity_score > threshold:
                    current_state = "STATE: ACTIVE"
                    color = (0, 0, 255) # Red text
                else:
                    current_state = "STATE: CALM"
                    color = (0, 255, 0) # Green text
                    
                # If state has changed, isolate and save the image!
                if current_state != previous_state and previous_state != "ANALYZING...":
                    clean_state_name = current_state.replace("STATE: ", "").replace("/", "_")
                    filename = f"{output_dir}/frame_{frame_count}_flipped_to_{clean_state_name}.jpg"
                    
                    export_frame = annotated_frame.copy()
                    cv2.putText(export_frame, "!!! STATE FLIP !!!", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imwrite(filename, export_frame)
                    print(f"State changed! Saved isolated frame: {filename}")
                    
                previous_state = current_state
                
                # Log the data for this frame to our CSV file
                csv_writer.writerow([frame_count, round(activity_score, 2), current_state.replace("STATE: ", "")])
                # Overlay the state and score on the video
                cv2.putText(annotated_frame, current_state, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                cv2.putText(annotated_frame, f"Activity Score: {activity_score:.1f}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add Jitter metric to UI for transparency
                cv2.putText(annotated_frame, f"Jitter Var: {total_jitter:.0f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Write to our video files
        out_normal.write(annotated_frame)
        out_slow.write(annotated_frame)

        cv2.imshow('Kinetic K9 - Phase 3: Pose Estimation', annotated_frame)
        frame_count += 1

        # Press 'q' to exit. Use 1ms delay to render the files as fast as the M3 chip allows!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Playback interrupted by user.")
            break

    # Clean up and neatly close the telemetry file
    csv_file.close()
    out_normal.release()
    out_slow.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Finished writing {out_normal_path} and {out_slow_path}.")

if __name__ == '__main__':
    # Default to 'dog_video.mov' in the current directory if no argument provided
    default_video = "dog_video.mov"
    
    if len(sys.argv) > 1:
        video_arg = sys.argv[1]
    else:
        video_arg = default_video

    play_video(video_arg)
