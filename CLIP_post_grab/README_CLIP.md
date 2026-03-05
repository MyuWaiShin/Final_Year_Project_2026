# CLIP for post-grab gripper verification

Use [CLIP](https://github.com/openai/CLIP) (Contrastive Language–Image Pre-training) to verify whether the gripper is **holding** an object or is **empty** after a pick attempt. This gives you a vision-based check to complement force + width (e.g. RG2 DI8/AI2) and reduce false positives.

---

## What the code is doing

1. **Load CLIP** — One neural net that encodes both images and text into the same “embedding” space (vectors). Similar meanings → vectors close together.
2. **Encode the prompts** — The 6 text strings (e.g. “a robot gripper holding an object…”, “an empty robotic gripper”…) are tokenized and encoded into 6 vectors. We keep these in memory.
3. **Encode your image** — Your gripper photo is preprocessed (resize, normalize) and encoded into one vector in that same space.
4. **Compare** — We compute similarity (dot product) between the image vector and each text vector. Higher = “this caption fits this image better”.
5. **Softmax** — Those similarities are turned into probabilities that sum to 1. The prompt with the highest probability wins.
6. **Label** — The winning prompt is either from the “holding” set or the “empty” set → we return `"holding"` or `"empty"`. If you use `--threshold` and the best probability is below it, we return `"uncertain"`.

So: **one image in → one label out** (“holding” / “empty” / “uncertain”) plus a confidence score.

---

## How fast is it? Can I get a result in 1–2 seconds?

**Yes.** Once the model is loaded, classifying one image is usually **well under 1 second** (often ~50–200 ms on CPU, ~10–50 ms on GPU). The script prints timing so you can see on your machine.

- **First run** (or every time you start the script): Loading the model can take a few seconds (e.g. 2–10 s). That’s a one-off cost per process.
- **Each extra image** (same process): Only the inference time above — so you can easily get “holding or empty?” in 1–2 seconds total, and often in a fraction of a second.

**For your robot:** Load the model once at startup (or in a long-lived service). Then each post-grab check is just one image → one quick inference, so you get an answer in under a second (or two on slow CPU). You don’t need to reload for every pick.

---

## What CLIP can do for you

- **Zero-shot**: No need to collect or label gripper images; you only define text descriptions.
- **Flexible prompts**: You can add more classes (e.g. "object slipping") or tweak wording for your camera angle and gripper type.
- **Single image in, label out**: Pass one image; get "holding", "empty", or "uncertain" (if using a confidence threshold).

**Limitations**: CLIP was trained on general internet images, not robot grippers. It can still work well for "holding" vs "empty" if the crop clearly shows the gripper; for best results use a consistent camera pose and crop. If accuracy is not enough, consider fine-tuning or a small custom classifier later.

## Setup

1. **Python**: 3.8+ recommended.

2. **Install PyTorch** (see [pytorch.org](https://pytorch.org) for your OS/CUDA):

   ```bash
   pip install torch torchvision
   ```

3. **Install OpenAI CLIP**:

   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

4. **Pillow** (if not already installed):

   ```bash
   pip install pillow
   ```

## How to test

1. **Install dependencies** (once):

   ```bash
   pip install torch torchvision pillow
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **Run on any image** to confirm the script works. Use a gripper photo if you have one, or any image (e.g. from your repo) to see scores:

   ```bash
   cd "c:\Users\myuwa\OneDrive - Middlesex University\Major Project\final-year-blog"
   python scripts/clip_gripper_verify.py assets/img/model2_yolov8n_train_stats.png
   ```

   If you don’t have a gripper image yet, you can use any photo (e.g. a random screenshot); the result will just show which prompt CLIP thinks fits best. For real verification you’ll want a crop of your RG2 gripper (with or without an object).

3. **See per-prompt scores** (useful to tune prompts):

   ```bash
   python scripts/clip_gripper_verify.py path/to/image.jpg --verbose
   ```

4. **Require a minimum confidence** so low-confidence predictions become `uncertain`:

   ```bash
   python scripts/clip_gripper_verify.py path/to/image.jpg --threshold 0.6
   ```

---

## Run inference without loading every time

The slow part (~6 s) is **loading the model**. Inference per image is ~400 ms. So you want to **load once**, then run inference many times.

### Option 1: Multiple images in one command

Pass several image paths in a single run. The model loads **once**, then each image is only inference:

```bash
python scripts/clip_gripper_verify.py assets/img/eye_in_hand_angle.png assets/img/another.png assets/img/third.png
```

You’ll see:
- One “Model loaded in X.XXs” at the start.
- For each image: “XXX ms -> 'holding' / 'empty' (confidence: …)”.

So after the first 6 s, every extra image is only ~400 ms.

### Option 2: From your own Python script (e.g. robot loop)

In your main program (e.g. pick-and-place loop), **load the model once at startup**, then call `predict()` each time you have a new gripper image. No reload.

```python
import torch
from clip_gripper_verify import load_model, get_prompt_embeddings, predict, PROMPTS

# --- Once at startup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_model(device)
prompts_flat, text_features = get_prompt_embeddings(
    model, [PROMPTS["holding"], PROMPTS["empty"]], device
)

# --- Every time you have a new gripper image (e.g. after each pick) ---
label, confidence, scores = predict(
    "path/to/gripper_capture.png",
    model, preprocess, device,
    prompts_flat, text_features,
    threshold=0.5,
)
# label is "holding", "empty", or "uncertain"
if label == "holding":
    # proceed to place
else:
    # retry or recover
```

Run your script once; each pick you only pay the inference time (~400 ms), not the 6 s load.

Make sure Python can find the module: run your script from the `scripts` folder, or from the repo root with `scripts` on `PYTHONPATH`, or add `sys.path` before importing:

```python
import sys
sys.path.insert(0, "scripts")  # or path to folder containing clip_gripper_verify.py
from clip_gripper_verify import load_model, get_prompt_embeddings, predict, PROMPTS
```

---

**Example output (single image):**

```
Using device: cuda
Loading CLIP (ViT-B/32)...

Result: 'holding' (confidence: 0.723)
```

With `--verbose` you also get something like:

```
Scores per prompt:
  0.723  a robot gripper holding an object between its fingers
  0.156  gripper with something in it
  ...
```

---

## Usage (reference)

From the repo root (or wherever the script lives):

```bash
python scripts/clip_gripper_verify.py path/to/your/gripper_image.jpg
```

**Options:**

- `--threshold 0.6` — If the best score is below 0.6, output is `uncertain`.
- `--verbose` — Print scores for each text prompt.
- `--device cuda` or `--device cpu` — Override device (default: auto).

## Integrating with your pipeline

1. After the arm closes the gripper and you have force/width “object detected”, capture a frame from your camera (wrist or overhead) and crop to the gripper region.
2. Run this script (or call `predict()` from your own code) on that image.
3. If result is `"holding"` and confidence is above your threshold, treat the pick as confirmed; if `"empty"` or `"uncertain"`, treat as failed and run recovery (e.g. retry or adjust pose).

You can import and use the functions directly:

```python
from clip_gripper_verify import load_model, get_prompt_embeddings, predict, PROMPTS

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_model(device)
prompts_flat, text_features = get_prompt_embeddings(
    model, [PROMPTS["holding"], PROMPTS["empty"]], device
)
label, confidence, scores = predict(
    "path/to/image.jpg", model, preprocess, device,
    prompts_flat, text_features, threshold=0.5
)
# label is "holding", "empty", or "uncertain"
```

## Customising prompts

Edit `PROMPTS` in `clip_gripper_verify.py` to add or change text descriptions. More prompts per class can improve robustness. Examples:

- **Holding**: "a robot gripper holding an object between its fingers", "gripper grasping a cube".
- **Empty**: "an empty robot gripper", "gripper with nothing in it".

Keep phrasing consistent with how your gripper looks in the image (e.g. "RG2 gripper" if that helps).

## Alternative: OpenCLIP

For more models and ongoing updates, you can use [OpenCLIP](https://github.com/mlfoundations/open_clip) (`open-clip-torch`). The API is similar: load model, encode image and text, compare similarities. The script in this folder uses the original OpenAI CLIP for simplicity.
