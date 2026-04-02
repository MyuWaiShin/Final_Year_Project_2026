import os
import glob
import pickle
import torch
import clip
from PIL import Image
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError:
    print("scikit-learn not found. Please run: pip install scikit-learn")
    import sys
    sys.exit(1)

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP (ViT-B/32) on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Directories
    base_dir    = r"C:\Users\myuwa\.gemini\antigravity\scratch\Final_Year_Project_2026\full_pipeline\temp\classification"
    holding_dir = os.path.join(base_dir, "holding")
    empty_dir   = os.path.join(base_dir, "empty")

    holding_images = glob.glob(os.path.join(holding_dir, "*.png")) + glob.glob(os.path.join(holding_dir, "*.jpg"))
    empty_images = glob.glob(os.path.join(empty_dir, "*.png")) + glob.glob(os.path.join(empty_dir, "*.jpg"))

    if len(holding_images) == 0 and len(empty_images) == 0:
        print("No images found in clip_dataset/cropped. Please collect data first.")
        return

    print(f"Found {len(holding_images)} holding images and {len(empty_images)} empty images.")
    if len(holding_images) < 2 or len(empty_images) < 2:
        print("\n[WARNING] You have very few images in one or more classes!")
        print("The model can still train, but please collect more images for a reliable classifier.")

    features = []
    labels = []

    print("\nExtracting CLIP features from images (this is fast)...")
    with torch.no_grad():
        for img_path in holding_images:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            feature = model.encode_image(image_input)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.cpu().numpy().flatten())
            labels.append(1)  # 1 = holding

        for img_path in empty_images:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            feature = model.encode_image(image_input)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.cpu().numpy().flatten())
            labels.append(0)  # 0 = empty

    X = np.array(features)
    y = np.array(labels)

    # We need to split into train and test so we can see real accuracy
    # But if there are less than 5 images per class, stratify will crash, so we handle that edge case
    if len(holding_images) >= 5 and len(empty_images) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        is_split = True
    else:
        print("\nDataset is too small to split safely. Training and testing on the entire dataset.")
        X_train, X_test = X, X
        y_train, y_test = y, y
        is_split = False

    print("\nTraining Logistic Regression classifier (Linear Probe)...")
    clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print(f"\n--- Evaluation Results ---")
    print(f"Training Accuracy: {accuracy_score(y_train, train_preds)*100:.1f}%")
    print(f"Testing Accuracy:  {accuracy_score(y_test, test_preds)*100:.1f}%")
    
    # Only print classification report if we have both classes present in the test set
    if len(np.unique(y_test)) > 1:
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_preds, target_names=["Empty", "Holding"]))
    
    if is_split:
        print("\nTraining a final model using ALL available data for maximum robustness...")
        clf.fit(X, y)

    # Save the trained model parameters to a file
    model_save_path = os.path.join(os.path.dirname(__file__), "clip_probe_v2.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"\nSaved trained classifier to -> {model_save_path}")
    print("You can now load this .pkl file during live robot execution for lightning-fast inference!")

if __name__ == "__main__":
    main()
