import torch
import clip
from PIL import Image
import pickle
import argparse
import sys
import os

def load_probe_classifier(pkl_path="clip_probe.pkl"):
    """Loads the trained Logistic Regression model."""
    if not os.path.exists(pkl_path):
        print(f"Error: Could not find '{pkl_path}'. Did you run train_clip_probe.py?")
        sys.exit(1)
        
    with open(pkl_path, "rb") as f:
        clf = pickle.load(f)
    return clf

def main():
    parser = argparse.ArgumentParser(description="Use the trained Linear Probe to classify gripper state.")
    parser.add_argument("image", type=str, help="Path to the gripper image crop")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    # 1. Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 2. Load our tiny trained classifier
    print("Loading custom trained Linear Probe...")
    clf = load_probe_classifier("clip_probe.pkl")

    # 3. Process the image
    image = Image.open(args.image).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 4. Extract features
    with torch.no_grad():
        feature = model.encode_image(image_input)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        # Convert to numpy block for scikit-learn
        feature_np = feature.cpu().numpy()

    # 5. Predict
    prediction = clf.predict(feature_np)[0]
    probabilities = clf.predict_proba(feature_np)[0]

    label_map = {0: "Empty", 1: "Holding"}
    predicted_label = label_map[prediction]
    
    # Probabilities usually come as [prob_class_0, prob_class_1]
    confidence = probabilities[prediction]

    print("\n" + "=" * 40)
    print(f"IMAGE: {args.image}")
    print(f"RESULT:  >> {predicted_label.upper()} <<")
    print(f"CONFIDENCE: {confidence * 100:.1f}%")
    print("=" * 40)

if __name__ == "__main__":
    main()
