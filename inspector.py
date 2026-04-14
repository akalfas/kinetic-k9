import cv2
import sys
import os

def play_video(path):
    if not os.path.exists(path):
        print(f"Error: Could not find '{path}'")
        return
        
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open '{path}'")
        return
        
    delay = 150  # Start very slow (150ms per frame)
    paused = False
    
    print("\n--- Kinetic K9 Inspector ---")
    print(f"Playing: {path}\n")
    print("Keyboard Controls:")
    print("  [SPACE] - Pause / Play")
    print("  [ f ]   - Step forward exactly 1 frame (when paused)")
    print("  [ - ]   - Slower (Increase delay)")
    print("  [ = ]   - Faster (Decrease delay)")
    print("  [ q ]   - Quit\n")
    
    ret, frame = cap.read()
    if not ret:
        print("Video is empty.")
        return
        
    while True:
        cv2.imshow("Kinetic K9 - Slow Motion Inspector (Focus this window!)", frame)
        
        # If paused, wait until input. If not, wait 'delay' ms.
        wait_time = 0 if paused else delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space
            paused = not paused
            state_str = "PAUSED" if paused else "PLAYING"
            print(f"[{state_str}]")
        elif key == ord('f') and paused:
            # Step one frame forward safely
            ret, next_frame = cap.read()
            if ret:
                frame = next_frame
            else:
                print("End of video reached.")
        elif key == ord('-'):
            delay = min(1000, delay + 25)
            print(f"Slowing down (Delay: {delay}ms)")
        elif key == ord('=') or key == ord('+'):
            delay = max(5, delay - 25)
            print(f"Speeding up (Delay: {delay}ms)")
            
        # Only fetch next frame if we are playing globally
        if not paused and key not in (ord('-'), ord('='), ord('+')):
            ret, next_frame = cap.read()
            if ret:
                frame = next_frame
            else:
                # Loop back or just break? Let's just pause at the end.
                print("End of video reached. Paused.")
                paused = True


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        play_video(sys.argv[1])
    else:
        # Default to the previously generated artifact
        fallback = "alf_1_anxiety_overlay.mp4"
        if os.path.exists(fallback):
            play_video(fallback)
        else:
            print("Please provide a video path: python inspector.py <video>")
