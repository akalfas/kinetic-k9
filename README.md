# Kinetic K9: Dual-Model Anxiety Engine

Computer Vision and Biometric logic engine designed to evaluate real-time canine micro-expressions—specifically capturing "Lip Licking", a dominant indicator of shifting anxiety states in dogs.

Currently engineered to operate locally on Apple Silicon (`MPS`) maintaining `<10ms` real-time execution speeds.

---

## 🛑 Phase 1: The Heuristic Dead End
Initially, the pipeline attempted to deduce lip-licking behavior strictly using **mathematical heuristics**. 
The logic was structured over a `YOLO26-Pose` model. We attempted to calculate the inter-frame velocity and vertical distance between the `Nose (16)` and `Chin (17)` keypoints.

### The Mathematical Failure: "The Jitter"
Upon deploying algorithmic telemetry, we computationally proved the system was physically destined to fail due to structural noise floors:
1. **The Signal:** The physical, real-world visual difference of a dog flicking out its tongue translates to roughly `10 to 15 pixels` of localized displacement.
2. **The Noise Floor:** The `YOLO26-Pose` spatial tracking algorithm intrinsically "jitters" by `±30 pixels` frame-to-frame, especially around soft structures like the jaw line.
3. **The Result:** Because the tracking hallucination (`30px`) is twice as volatile as the physical biological movement (`15px`), mathematical calculation was rendered completely impossible. The engine inherently flagged `215+ False Positives` from the dog merely breathing or turning its head.

## 🔥 Phase 2: The Architectural Solution (Dual-Model Pipeline)
To bypass the mathematical noise floor entirely, we abandoned algorithmic calculations and introduced a **Dual-Model** structured architecture.

We paired the heavy Pose Estimator with an ultra-lightweight, highly concentrated Nano Image Classifier (`YOLO11n-cls`).

### The Inference Execution Loop:
1. **Coordinate Locking:** The primary `YOLO26-Pose` model locks onto the face architecture. We extract **only** the `Nose (16)` center-point.
2. **Dynamic Muzzle Slicing:** Instead of calculating distances, the engine violently array-slices a highly stabilized `160x160` geometric bounding box wrapped securely around the muzzle, intentionally excluding the eyes and ears.
3. **Secondary Neural Gate:** This hyper-local crop is instantaneously passed into the Nano Image Classifier operating at FP16 half-precision, which relies purely on structural CNN texture map recognition to identify the exact visualization of a tongue resting over the lips.

### The Biological Duration Filter & Synthesis
Because the neural network is extremely sensitive, shadow movement on the upper lip caused single-frame spikes (`>95%` certainty) mimicking tongue shapes.

To physically perfect the false-positive rejection, we introduced:
* **The Biological Duration Gate:** A physical lip lick takes roughly ~30 frames. The pipeline enforces a rigid `6-frame consecutive threshold`. Any 1-frame or 3-frame "hallucinations" from the CNN are immediately rejected, while massive block licks completely pass the gate.
* **Synthetic Overcompensation:** The dataset contains synthetic Gen-AI injected grid grids (via Gemini Imagen prompts) to force the Nano Classifier to deeply generalize across breeds, rather than overfitting onto a single dog's ambient lighting. 

---

## 🛠 Repository Structure

```text
├── src/
│   ├── main.py                  # Standard pipeline runner
│   ├── anxiety_engine.py        # Core algorithm & Biological gating constraints
│   └── render_anxiety.py        # High-fidelity Video UI Overlay generator
├── tools/
│   ├── build_mouth_dataset.py   # Uses YOLO-Pose to dynamically slice local crops
│   ├── train_mouth_cls.py       # Executes CNN training loop on Apple MPS
│   └── process_dataset_grids.py # Synthetic augmentation tool
```

## Execute the Engine

To deploy the UI overlay visualizer across raw media files:
```bash
python src/render_anxiety.py
```
*(Note: Large video assets and raw `.pt` weights are omitted from this Git repository via `.gitignore` to maintain server integrity. Models run strictly locally.)*
