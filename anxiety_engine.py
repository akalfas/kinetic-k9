import numpy as np
from collections import deque
from ultralytics import YOLO

class YOLO26AnxietyEngine:
    def __init__(self, model_path="runs/pose/training_run_1/weights/best.pt", device="mps"):
        print(f"Loading YOLO26 Anxiety Engine with model: {model_path}")
        self.model = YOLO(model_path).to(device)
        self.mouth_cls = YOLO("runs/classify/runs/cls/mouth_run2/weights/best.pt").to(device)
        
        # Hyperparameters
        self.history_size = 30
        self.ema_alpha = 0.15 # For Temporal Smoothing
        
        # State tracking
        self.nose_y_history = deque(maxlen=self.history_size)
        self.baseline_length = None
        self.calibration_lengths = []
        self.smoothed_anxiety_score = 0.0
        self.lick_duration = 0
        
        # Keypoint Mapping
        self.KP = {
            "Nose": 0, "L_Eye": 1, "R_Eye": 2, "L_Ear_Base": 3, "R_Ear_Base": 4, 
            "L_Ear_Tip": 5, "R_Ear_Tip": 6, "Neck": 7, "Mid_Back": 8,
            "Shoulder": 10, "Tail_Base": 22, "Tail_Tip": 23
        }
        
    def _get_valid_pt(self, kpts, idx_list):
        """Returns the average of valid keypoints from the list (if they exist)."""
        valid = [kpts[i] for i in idx_list if kpts[i][0] > 0 and kpts[i][1] > 0]
        if len(valid) == 0:
            return None
        return np.mean(valid, axis=0)

    def process_frame(self, frame):
        """
        Process a single frame to calculate the anxiety score.
        Uses max_det=1 and disables NMS for latency optimization.
        """
        # 1. NMS-free Inference
        # In fully compliant YOLO26 we'd pass agnostic_nms=False or utilize raw tensor hook
        results = self.model.predict(
            frame, 
            verbose=False, 
            max_det=1, 
            half=True,       # FP16 quantization for raw speed
            conf=0.25        # Early drop of low conf
        )
        
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return self.smoothed_anxiety_score, "NO_DETECTION", {}

        # Slicing the raw numpy tensor for vectorization
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        # Guard against zero-detected poses
        if len(kpts) < 24:
            return self.smoothed_anxiety_score, "PARTIAL_DETECTION", {}

        # Extract features
        nose = self._get_valid_pt(kpts, [self.KP["Nose"]])
        neck = self._get_valid_pt(kpts, [self.KP["Neck"]])
        shoulder = self._get_valid_pt(kpts, [self.KP["Shoulder"]])
        mid_back = self._get_valid_pt(kpts, [self.KP["Mid_Back"]])
        tail_base = self._get_valid_pt(kpts, [self.KP["Tail_Base"]])
        tail_tip = self._get_valid_pt(kpts, [self.KP["Tail_Tip"]])
        ear_tips = self._get_valid_pt(kpts, [self.KP["L_Ear_Tip"], self.KP["R_Ear_Tip"]])
        
        metrics = {}
        raw_anxiety_score = 0.0
        
        # 2. Ear-to-Tail Tension
        tension_ratio = 1.0
        metrics["ears_pinned"] = False
        if ear_tips is not None and tail_base is not None:
            dist = np.linalg.norm(ear_tips - tail_base)
            
            # Calibration Phase: Establish baseline distance
            if self.baseline_length is None:
                self.calibration_lengths.append(dist)
                if len(self.calibration_lengths) >= self.history_size:
                    # Take the 75th percentile as the "relaxed" length
                    self.baseline_length = np.percentile(self.calibration_lengths, 75)
            else:
                tension_ratio = dist / self.baseline_length
                metrics["ear_tail_ratio"] = tension_ratio
                
                # Significant drop signals pinned ears / compressed posture
                if tension_ratio < 0.75:
                    metrics["ears_pinned"] = True
                    raw_anxiety_score += 0.3
                    
        # 3. Head/Shoulder Ratio ("Cowering")
        metrics["cowering"] = False
        if nose is not None and neck is not None and shoulder is not None:
            # In image coords, larger Y is lower physically.
            drop_nose = nose[1] > shoulder[1]
            drop_neck = neck[1] > shoulder[1]
            
            if drop_nose and drop_neck:
                metrics["cowering"] = True
                raw_anxiety_score += 0.35
                
        # 4. Tail Tuck Angle
        metrics["tail_tucked"] = False
        if mid_back is not None and tail_base is not None and tail_tip is not None:
            # Unit vectors
            spine_vec = tail_base - mid_back
            spine_len = np.linalg.norm(spine_vec)
            
            tail_vec = tail_tip - tail_base
            tail_len = np.linalg.norm(tail_vec)
            
            if spine_len > 0 and tail_len > 0:
                spine_u = spine_vec / spine_len
                tail_u = tail_vec / tail_len
                
                # Dot product
                dot_prod = np.clip(np.dot(spine_u, tail_u), -1.0, 1.0)
                angle_rad = np.arccos(dot_prod)
                angle_deg = np.degrees(angle_rad)
                metrics["tail_angle"] = angle_deg
                
                # If angle is large, tail is tucked aggressively back under the body line
                if angle_deg > 110:
                    metrics["tail_tucked"] = True
                    raw_anxiety_score += 0.35
                    
        # 5. Dedicated CNN Lip Lick Detection
        metrics["licking"] = False
        kpts_full = results[0].keypoints.xy[0].cpu().numpy()
        nose_x, nose_y = kpts_full[16][0], kpts_full[16][1]
        
        is_lick_frame = False
        
        if nose_x > 0 and nose_y > 0:
            # Replicate training crop parameters identically
            y1 = int(max(0, nose_y - 40))
            y2 = int(min(frame.shape[0], nose_y + 120))
            x1 = int(max(0, nose_x - 80))
            x2 = int(min(frame.shape[1], nose_x + 80))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 50 and crop.shape[1] > 50:
                cls_results = self.mouth_cls.predict(crop, verbose=False, half=True)
                
                if len(cls_results) > 0:
                    names_dict = cls_results[0].names
                    probs = cls_results[0].probs
                    
                    if "licking" in names_dict.values():
                        lick_idx = list(names_dict.values()).index("licking")
                        lick_prob = probs.data[lick_idx].item()
                        
                        # 85% Confidence Threshold for structural lick
                        if lick_prob > 0.85:
                            is_lick_frame = True
                            
        if is_lick_frame:
            self.lick_duration += 1
        else:
            self.lick_duration = 0
            
        # 6-Frame Biological Gate: A real dog tongue movement takes ~30 frames.
        # This completely filters out 1-frame shadow hallucinations.
        if self.lick_duration >= 6:
            metrics["licking"] = True
            raw_anxiety_score += 0.15
                    
        # Cap Raw Score
        raw_anxiety_score = min(raw_anxiety_score, 1.0)
        
        # 6. Temporal Smoothing (EMA)
        self.smoothed_anxiety_score = (self.ema_alpha * raw_anxiety_score) + \
                                      ((1.0 - self.ema_alpha) * self.smoothed_anxiety_score)
                                      
        # Determine strict categorical State
        state = "CALM"
        if self.smoothed_anxiety_score > 0.6:
            state = "ANXIOUS_SEVERE"
        elif self.smoothed_anxiety_score > 0.3:
            state = "ANXIOUS_MILD"
            
        return self.smoothed_anxiety_score, state, metrics
