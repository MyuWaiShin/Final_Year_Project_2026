#!/usr/bin/env python3
"""
Post-grab verification using CLIP: classify a gripper image as "holding" or "empty".

Use after the arm has attempted a pick; pass a cropped/framed image of the gripper
(e.g. from a wrist or overhead camera). CLIP scores the image against text prompts
and returns which description fits best.

WHAT THE CODE DOES (high level):
  1. Load CLIP model (image + text encoder).
  2. Encode 6 text prompts (3 for "holding", 3 for "empty") into vectors.
  3. Load your image, encode it into a vector in the same space.
  4. Compare image vector to each text vector (similarity); softmax → probabilities.
  5. The prompt with highest probability wins → we map that back to "holding" or "empty".
  6. If you set --threshold and max prob is below it, we return "uncertain".

Requirements:
    pip install torch torchvision
    pip install git+https://github.com/openai/CLIP.git

Usage:
    # One image (loads model, runs inference, exits — model is loaded every run)
    python clip_gripper_verify.py path/to/image.jpg

    # Multiple images in one run: model loads ONCE, then inference for each image (~400ms each)
    python clip_gripper_verify.py img1.png img2.png img3.png

    # From your own code: load once at startup, then call predict() in a loop (no reload)
    python clip_gripper_verify.py path/to/image.jpg --threshold 0.6 --verbose
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch

try:
    import clip
    from PIL import Image
except ImportError as e:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "  pip install torch torchvision pillow\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    ) from e


# Default text prompts for gripper state (customize for your setup)
PROMPTS = {
    "holding": [
        "a robot gripper holding an object between its fingers",
        "a robotic hand grasping an object",
        "gripper with something in it",
    ],
    "empty": [
        "a robot gripper with nothing in it",
        "an empty robotic gripper",
        "gripper with no object",
    ],
}


def load_model(device: str):
    """Load CLIP model and preprocessing (ViT-B/32 is a good default)."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def get_prompt_embeddings(model, text_prompts: list[str], device: str):
    """Tokenize and encode all prompts; return (tokens, features)."""
    flat = [p for prompts in text_prompts for p in prompts]
    tokens = clip.tokenize(flat, truncate=True).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    return flat, features


def predict(
    image_path: str,
    model,
    preprocess,
    device: str,
    prompts_flat: list[str],
    text_features: torch.Tensor,
    threshold: Optional[float] = None,
):
    """
    Run zero-shot classification: score image against each prompt and pick best.
    Returns ("holding" | "empty", confidence, all_scores).
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image to a vector; normalize so we can use cosine similarity
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_n = text_features / text_features.norm(dim=-1, keepdim=True)
        # Similarity = dot product of normalized vectors; scale by 100 (CLIP temperature)
        logits = (100.0 * image_features @ text_features_n.T).squeeze(0)

    # Turn logits into probabilities (sum to 1); pick the prompt with highest prob
    probs = logits.softmax(dim=-1).cpu().numpy()
    best_idx = int(probs.argmax())
    best_prompt = prompts_flat[best_idx]
    confidence = float(probs[best_idx])

    n_holding = len(PROMPTS["holding"])
    label = "holding" if best_idx < n_holding else "empty"

    # Optional: require minimum confidence
    if threshold is not None and confidence < threshold:
        label = "uncertain"

    scores = dict(zip(prompts_flat, probs.tolist()))
    return label, confidence, scores


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based gripper verification: holding vs empty"
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="+",
        help="Path(s) to gripper image(s). One run = one model load; multiple paths = inference only for the rest.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, output 'uncertain' when max confidence < threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, or leave unset for auto",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-prompt scores",
    )
    args = parser.parse_args()

    images = [Path(p) for p in args.image]
    for p in images:
        if not p.exists():
            raise SystemExit(f"Image not found: {p}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    t0 = time.perf_counter()
    print("Loading CLIP (ViT-B/32)...")
    model, preprocess = load_model(device)
    prompts_flat, text_features = get_prompt_embeddings(
        model, [PROMPTS["holding"], PROMPTS["empty"]], device
    )
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.2f}s (only once for this run)\n")

    for i, path in enumerate(images):
        t1 = time.perf_counter()
        label, confidence, scores = predict(
            str(path), model, preprocess, device, prompts_flat, text_features, args.threshold
        )
        infer_s = time.perf_counter() - t1
        print(f"[{path.name}] {infer_s*1000:.0f} ms  ->  {label!r} (confidence: {confidence:.3f})")
        if args.verbose:
            for prompt, score in sorted(scores.items(), key=lambda x: -x[1]):
                print(f"    {score:.3f}  {prompt}")
            if i < len(images) - 1:
                print()


if __name__ == "__main__":
    main()
