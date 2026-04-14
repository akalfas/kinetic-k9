import cv2
import sys
import os
import numpy as np
from collections import deque
from anxiety_engine import YOLO26AnxietyEngine

def draw_anxiety_graph(frame, history_queue, max_len=200):
    """Draws a semi-transparent HUD graph of the anxiety score with event markers."""
    if len(history_queue) < 2:
        return frame
        
    h, w, _ = frame.shape
    graph_h = 100
    graph_w = 400
    margin = 30
    
    start_x = w - graph_w - margin
    start_y = h - graph_h - margin
    
    # Overlay semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + graph_w, start_y + graph_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Border & HUD Text
    cv2.rectangle(frame, (start_x, start_y), (start_x + graph_w, start_y + graph_h), (255, 255, 255), 1)
    
    # Legend
    cv2.putText(frame, "Anxiety Level History", (start_x + 5, start_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "| Licking Event", (start_x + graph_w - 120, start_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
    # Draw Graph Baseline
    cv2.line(frame, (start_x, start_y + graph_h), (start_x + graph_w, start_y + graph_h), (100, 100, 100), 1)
    
    # Calculate Line Points
    pts = []
    x_step = graph_w / max_len
    
    for i, data in enumerate(history_queue):
        score, is_licking = data
        
        # Clip score between 0.0 and 1.0 just in case
        clamped_score = max(0.0, min(1.0, score))
        x = int(start_x + (i * x_step))
        y = int((start_y + graph_h) - (clamped_score * graph_h))
        pts.append((x, y, is_licking))
        
    # Draw PolyLines and Markers
    for i in range(1, len(pts)):
        x1, y1, lick1 = pts[i-1]
        x2, y2, lick2 = pts[i]
        
        # Color gradient logic based on height
        color = (0, 255, 0) # Green if low anxiety
        if y2 < start_y + (graph_h * 0.7):
            color = (0, 165, 255) # Orange
        if y2 < start_y + (graph_h * 0.4):
            color = (0, 0, 255) # Red if high anxiety
            
        # Draw the graph line
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # Event Marker: Lip Licking
        if lick2:
            # Draw a hot pink vertical bar spanning the entire graph height
            cv2.line(frame, (x2, start_y), (x2, start_y + graph_h), (255, 0, 255), 2)
        
    return frame

def process_video(video_path, engine):
    print(f"\n--- Processing {video_path} ---")
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_name = os.path.splitext(video_path)[0]
    out_path = f"{base_name}_anxiety_overlay.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # History queue for the Graph now holds (score, is_licking)
    max_history_len = 200
    score_history = deque(maxlen=max_history_len)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = frame.copy()
        score, state, metrics = engine.process_frame(frame)
        
        is_licking = metrics.get('licking', False) if metrics else False
        score_history.append((score, is_licking))
        
        # Draw dynamic graph with licking pip markers
        annotated_frame = draw_anxiety_graph(annotated_frame, score_history, max_len=max_history_len)
        
        # Color mapping based on state
        if state == "ANXIOUS_SEVERE":
            color = (0, 0, 255) # Red
        elif state == "ANXIOUS_MILD":
            color = (0, 165, 255) # Orange
        elif state == "CALM":
            color = (0, 255, 0) # Green
        else:
            color = (150, 150, 150) # Gray for NO_DETECTION
            
        # Draw Master State
        cv2.putText(annotated_frame, f"STATE: {state}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        cv2.putText(annotated_frame, f"Anxiety Score: {score:.2f}", (30, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Draw sub-metrics
        y_offset = 200
        if metrics:
            for key, val in metrics.items():
                if isinstance(val, (float, np.floating)):
                    text = f"{key}: {val:.2f}"
                else:
                    text = f"{key}: {val}"
                    
                metric_color = (255, 255, 255)
                if val is True: 
                    metric_color = (0, 0, 255) # Warning red for active metrics in HUD
                    
                cv2.putText(annotated_frame, text, (30, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, metric_color, 2)
                y_offset += 40

        out.write(annotated_frame)
        
        # Real-time display
        cv2.imshow(f"Kinetic K9: {os.path.basename(video_path)}", annotated_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):  
            print("Playback skipped by user.")
            break
            
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved optimized output to {out_path}")

def main():
    model_path = "runs/pose/training_run_1/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo26m-pose.pt"
        
    print("Loading YOLO Engine Model...")
    engine = YOLO26AnxietyEngine(model_path=model_path)
    
    videos = ["alf_1.mov", "alf_2.mp4"]
    for video in videos:
        process_video(video, engine)
        
    print("\nAll processing complete. You can now view the isolated overlay videos!")

if __name__ == "__main__":
    main()
