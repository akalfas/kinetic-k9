import cv2
from anxiety_engine import YOLO26AnxietyEngine

def main():
    engine = YOLO26AnxietyEngine()
    cap = cv2.VideoCapture("alf_2.mp4")
    f_idx = 0
    
    # We will log any frame where lick probability is above 0.50 so we can see the sensitivity curve!
    print("Scanning alf_2.mp4 for Licking Probability > 0.50...")
    print("-" * 50)
    print(f"{'Frame':<8} | {'Prob':<8} | {'State'}")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Fast-forward disabled: scanning every frame for full-video telemetry
        if True:
            # We must hook into the underlying logic to get the raw float probability!
            kpts_full = engine.model.predict(frame, verbose=False, max_det=1, half=True)[0].keypoints
            if kpts_full is not None and len(kpts_full.xy) > 0:
                kpts = kpts_full.xy[0].cpu().numpy()
                nose_x, nose_y = kpts[16][0], kpts[16][1]
                if nose_x > 0 and nose_y > 0:
                    y1 = int(max(0, nose_y - 40))
                    y2 = int(min(frame.shape[0], nose_y + 120))
                    x1 = int(max(0, nose_x - 80))
                    x2 = int(min(frame.shape[1], nose_x + 80))
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 50 and crop.shape[1] > 50:
                        cls_res = engine.mouth_cls.predict(crop, verbose=False, half=True)
                        if len(cls_res) > 0:
                            names = cls_res[0].names
                            if "licking" in names.values():
                                lick_idx = list(names.values()).index("licking")
                                val = cls_res[0].probs.data[lick_idx].item()
                                
                                if val > 0.40:
                                    status = "TRIGGERED" if val > 0.85 else "below_threshold"
                                    print(f"{f_idx:<8d} | {val:<8.3f} | {status}")
        f_idx += 1
    cap.release()

if __name__ == "__main__":
    main()
