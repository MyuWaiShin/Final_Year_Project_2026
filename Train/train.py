"""
YOLO Training — v5n, v6n, v8n, v26n
Capstone Project: Grasp Failure Detection
Myu M00964135

DEPLOYMENT TARGETS:
    YOLOv5n  → OAK-D blob (best tested compatibility)
    YOLOv6n  → OAK-D blob (also supported)
    YOLOv8n  → OAK-D blob (supported, needs opset 12)
    YOLO26n  → Laptop CPU + Raspberry Pi (ONNX/OpenVINO only, no blob yet)

INSTALL:
    pip install ultralytics --upgrade   # v8 + v26
    pip install yolov5                  # v5
    git clone https://github.com/meituan/YOLOv6  # v6
    pip install onnx onnxruntime

DATA.YAML:
    path: C:/Users/myuwa/OneDrive/your_dataset
    train: images/train
    val:   images/val
    test:  images/test
    nc: 3
    names: ['cube', 'cylinder', 'arc']
"""

import os
import sys
import subprocess

# ─────────────────────────────────────────────
# CHANGE THESE
# ─────────────────────────────────────────────
DATA_YAML   = r"c:\Users\MS3433\Final_Year_Project_2026\Train\data.yaml"
PROJECT_DIR = "runs/train"
IMG_SIZE    = 640
BATCH       = 32        # RTX 5080 can handle a larger batch size
EPOCHS      = 100
PATIENCE    = 20
DEVICE      = 0         # 0 = GPU, 'cpu' = CPU only

# ─────────────────────────────────────────────
# SHARED AUGMENTATION
# ─────────────────────────────────────────────
AUG = dict(
    degrees     = 45,
    translate   = 0.2,
    scale       = 0.5,
    fliplr      = 0.5,
    flipud      = 0.1,
    perspective = 0.001,
    hsv_h       = 0.1,
    hsv_s       = 0.4,
    hsv_v       = 0.3,
    mosaic      = 1.0,
    mixup       = 0.05,
    erasing     = 0.1,
)

# YOLOv5 uses a hyp yaml file instead of kwargs
V5_HYP = """
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.1
hsv_s: 0.4
hsv_v: 0.3
degrees: 45.0
translate: 0.2
scale: 0.5
shear: 2.0
perspective: 0.001
flipud: 0.1
fliplr: 0.5
mosaic: 1.0
mixup: 0.05
copy_paste: 0.0
"""


# ═════════════════════════════════════════════
# YOLOv8n
# ═════════════════════════════════════════════
def train_v8():
    from ultralytics import YOLO
    print("\n" + "="*55)
    print("TRAINING YOLOv8n")
    print("="*55)

    model = YOLO('yolov8n.pt')
    model.train(
        data=DATA_YAML, epochs=EPOCHS, patience=PATIENCE,
        imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
        project=PROJECT_DIR, name='yolov8n', exist_ok=True,
        **AUG
    )
    print(f"✓ YOLOv8n done → {PROJECT_DIR}/yolov8n/weights/best.pt")


# ═════════════════════════════════════════════
# YOLO26n
# ═════════════════════════════════════════════
def train_v26():
    from ultralytics import YOLO
    print("\n" + "="*55)
    print("TRAINING YOLO26n")
    print("="*55)

    model = YOLO('yolo26n.pt')
    model.train(
        data=DATA_YAML, epochs=EPOCHS, patience=PATIENCE,
        imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
        project=PROJECT_DIR, name='yolo26n', exist_ok=True,
        **AUG
    )
    print(f"✓ YOLO26n done → {PROJECT_DIR}/yolo26n/weights/best.pt")


# ═════════════════════════════════════════════
# YOLOv5n
# ═════════════════════════════════════════════
def train_v5():
    print("\n" + "="*55)
    print("TRAINING YOLOv5n")
    print("="*55)

    with open('hyp_v5.yaml', 'w') as f:
        f.write(V5_HYP)

    cmd = [
        sys.executable, '-m', 'yolov5.train',
        '--data',     DATA_YAML,
        '--weights',  'yolov5n.pt',
        '--epochs',   str(EPOCHS),
        '--patience', str(PATIENCE),
        '--imgsz',    str(IMG_SIZE),
        '--batch',    str(BATCH),
        '--project',  PROJECT_DIR,
        '--name',     'yolov5n',
        '--hyp',      'hyp_v5.yaml',
        '--exist-ok',
    ]
    if DEVICE != 'cpu':
        cmd += ['--device', str(DEVICE)]

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"✓ YOLOv5n done → {PROJECT_DIR}/yolov5n/weights/best.pt")
    else:
        print("✗ YOLOv5n failed. Try manually:")
        print(f"  python -m yolov5.train --data {DATA_YAML} --weights yolov5n.pt --epochs {EPOCHS}")


# ═════════════════════════════════════════════
# YOLOv6n
# ═════════════════════════════════════════════
def train_v6():
    print("\n" + "="*55)
    print("TRAINING YOLOv6n")
    print("="*55)

    if not os.path.exists('YOLOv6'):
        print("Cloning YOLOv6 repo...")
        subprocess.run(['git', 'clone', 'https://github.com/meituan/YOLOv6'])
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        '-r', 'YOLOv6/requirements.txt'])

    os.makedirs('YOLOv6/configs/custom', exist_ok=True)
    cfg = """_base_ = '../yolov6n_finetune.py'
model = dict(nc=2)
data_aug = dict(
    hsv_h=0.1, hsv_s=0.4, hsv_v=0.3,
    degrees=45, translate=0.2, scale=0.5,
    flipud=0.1, fliplr=0.5,
    mosaic=1.0, mixup=0.05,
)
"""
    with open('YOLOv6/configs/custom/yolov6n_grasp.py', 'w') as f:
        f.write(cfg)

    cmd = [
        sys.executable, 'YOLOv6/tools/train.py',
        '--data-path',  DATA_YAML,
        '--conf',       'YOLOv6/configs/yolov6n_finetune.py',
        '--epochs',     str(EPOCHS),
        '--img-size',   str(IMG_SIZE),
        '--batch-size', str(BATCH),
        '--output-dir', f'{PROJECT_DIR}/yolov6n',
        '--name',       'yolov6n',
    ]
    if DEVICE != 'cpu':
        cmd += ['--device', str(DEVICE)]

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"✓ YOLOv6n done → {PROJECT_DIR}/yolov6n/")
    else:
        print("✗ YOLOv6n failed. Check YOLOv6 repo cloned correctly.")
        print("  See: https://github.com/meituan/YOLOv6")


# ═════════════════════════════════════════════
# EXPORT ALL TO ONNX
# ═════════════════════════════════════════════
def export_all():
    from ultralytics import YOLO

    print("\n" + "="*55)
    print("EXPORTING ALL MODELS TO ONNX")
    print("="*55)

    for name in ['yolov8n', 'yolo26n']:
        pt = f"{PROJECT_DIR}/{name}/weights/best.pt"
        if os.path.exists(pt):
            model = YOLO(pt)
            model.export(format='onnx', imgsz=IMG_SIZE, opset=12, simplify=True)
            print(f"✓ {name} → {pt.replace('.pt', '.onnx')}")
        else:
            print(f"✗ {name} not found, skipping")

    pt5 = f"{PROJECT_DIR}/yolov5n/weights/best.pt"
    if os.path.exists(pt5):
        subprocess.run([
            sys.executable, '-m', 'yolov5.export',
            '--weights', pt5, '--include', 'onnx',
            '--imgsz', str(IMG_SIZE), '--opset', '12', '--simplify',
        ])
        print(f"✓ yolov5n → {pt5.replace('.pt', '.onnx')}")
    else:
        print("✗ yolov5n not found, skipping")

    print("\n" + "-"*55)
    print("BLOB CONVERSION (OAK-D) — go to blobconverter.luxonis.com")
    print("  Upload:   best.onnx for v5n, v6n, or v8n")
    print("  Settings: Shaves=6, FP16, RVC2")
    print("  NOTE:     YOLO26n blob NOT supported yet — use ONNX for CPU/Pi")


# ═════════════════════════════════════════════
# COMPARE RESULTS
# ═════════════════════════════════════════════
def compare():
    import csv

    print("\n" + "="*60)
    print(f"{'Model':<12} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'OAK-D blob':>12}")
    print("-"*60)

    models = {
        'YOLOv5n' : (f"{PROJECT_DIR}/yolov5n/results.csv",  "✓ yes"),
        'YOLOv6n' : (f"{PROJECT_DIR}/yolov6n/results.csv",  "✓ yes"),
        'YOLOv8n' : (f"{PROJECT_DIR}/yolov8n/results.csv",  "✓ yes"),
        'YOLO26n' : (f"{PROJECT_DIR}/yolo26n/results.csv",  "✗ not yet"),
    }

    for name, (path, blob) in models.items():
        if not os.path.exists(path):
            print(f"{name:<12} {'not trained yet':<26} {blob:>12}")
            continue
        with open(path) as f:
            rows = [{k.strip(): v.strip() for k,v in r.items()}
                    for r in csv.DictReader(f)]
        if not rows:
            continue
        best    = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0) or 0))
        map50   = best.get('metrics/mAP50(B)',    'N/A')
        map5095 = best.get('metrics/mAP50-95(B)', 'N/A')
        print(f"{name:<12} {map50:>10} {map5095:>14} {blob:>12}")

    print("\nRECOMMENDATION:")
    print("  Best mAP with blob support → deploy on OAK-D")
    print("  YOLO26n ONNX               → deploy on CPU/Pi")


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
        choices=['v5','v6','v8','v26','all'], default='all')
    parser.add_argument('--export',  action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    if args.train == 'all':
        train_v8()
        train_v26()
        train_v5()
        train_v6()
    elif args.train == 'v8':  train_v8()
    elif args.train == 'v26': train_v26()
    elif args.train == 'v5':  train_v5()
    elif args.train == 'v6':  train_v6()

    if args.export:
        export_all()

    if args.compare:
        compare()

    print("\nUSAGE:")
    print("  python train.py                              # train all 4")
    print("  python train.py --train v8                  # train one")
    print("  python train.py --train all --export        # train + export")
    print("  python train.py --compare                   # compare results")
    print("  python train.py --train all --export --compare  # full pipeline")