import cv2
from anxiety_engine import YOLO26AnxietyEngine

def test_final_accuracy():
    print("Initializing Dual-Model Engine (YOLO-Pose + YOLO-Mouth-Cls)...")
    engine = YOLO26AnxietyEngine()
    
    cap = cv2.VideoCapture("alf_2.mp4")
    f_idx = 0
    licking_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Only process frames we care about to speed up the test
        if 200 <= f_idx <= 300:
            score, state, metrics = engine.process_frame(frame)
            if metrics.get("licking", False):
                licking_frames.append(f_idx)
                
        f_idx += 1
        
    cap.release()
    
    print("\n--- TEST RESULTS ---")
    print(f"Total Licking Frames Detected: {len(licking_frames)}")
    if licking_frames:
        # Group sequential frames for easy reading
        ranges = []
        start = licking_frames[0]
        prev = start
        for f in licking_frames[1:]:
            if f == prev + 1:
                prev = f
            else:
                ranges.append(f"{start}-{prev}")
                start = f
                prev = f
        ranges.append(f"{start}-{prev}")
        print("Detected Lip Licks exactly spanning:")
        for r in ranges:
            print(f"  > Frames {r}")
    
    print("\nComparison to Ground Truth:")
    print("  > True Lick 1: 236-271")
    print("  > True Lick 2: 279-286")

if __name__ == "__main__":
    test_final_accuracy()
