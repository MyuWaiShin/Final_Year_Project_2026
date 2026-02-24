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
import contextlib

# Fix Windows console encoding for checkpoints and emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

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

# Git absolute path for this machine
GIT_EXE = r"C:\Users\MS3433\AppData\Local\Programs\Git\cmd\git.exe"

# Configure environment for GitPython and Subprocesses
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = GIT_EXE
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
git_dir = os.path.dirname(GIT_EXE)
if git_dir not in os.environ['PATH']:
    os.environ['PATH'] = git_dir + os.pathsep + os.environ['PATH']

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

    # Auto-cloning yolov5 if missing
    if not os.path.exists('yolov5'):
        print("Cloning YOLOv5 repo...")
        subprocess.run([GIT_EXE, 'clone', 'https://github.com/ultralytics/yolov5'])
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'yolov5/requirements.txt'])

    if os.path.abspath('yolov5') not in sys.path:
        sys.path.append(os.path.abspath('yolov5'))

    cmd = [
        sys.executable, 'yolov5/train.py',
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
        subprocess.run([GIT_EXE, 'clone', 'https://github.com/meituan/YOLOv6'])
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        '-r', 'YOLOv6/requirements.txt'])

    if os.path.abspath('YOLOv6') not in sys.path:
        sys.path.append(os.path.abspath('YOLOv6'))

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

    # Set PYTHONPATH so yolov6 can find itself
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('YOLOv6') + os.pathsep + env.get('PYTHONPATH', '')

    cmd = [
        sys.executable, 'tools/train.py',
        '--data-path',  os.path.abspath(DATA_YAML),
        '--conf',       'configs/custom/yolov6n_grasp.py',
        '--epochs',     str(EPOCHS),
        '--img-size',   str(IMG_SIZE),
        '--batch-size', str(BATCH),
        '--output-dir', os.path.abspath(f'{PROJECT_DIR}/yolov6n'),
        '--name',       'yolov6n',
    ]
    if DEVICE != 'cpu':
        cmd += ['--device', str(DEVICE)]

    result = subprocess.run(cmd, cwd='YOLOv6', env=env)
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
        # Using yolov5/export.py directly to ensure params are handled
        subprocess.run([
            sys.executable, 'yolov5/export.py',
            '--weights', pt5, '--include', 'onnx',
            '--imgsz', str(IMG_SIZE), '--opset', '12'
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
def run_val_if_missing(model_name, pt_path):
    """Run validation via subprocess if results.csv is missing to avoid import conflicts."""
    print(f"  [i] Running quick validation to get metrics for {model_name}...")
    import json
    
    val_dir = f"{PROJECT_DIR}/{model_name}_val"
    os.makedirs(val_dir, exist_ok=True)
    
    # Run yolo val in a subprocess to avoid sys.path pollution from yolov5
    cmd = [
        sys.executable, '-m', 'ultralytics.cfg', 'val',
        f'model={os.path.abspath(pt_path)}',
        f'data={os.path.abspath(DATA_YAML)}',
        f'imgsz={IMG_SIZE}',
        f'project={os.path.abspath(PROJECT_DIR)}',
        f'name={model_name}_val',
        'exist_ok=True'
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # YOLO validation outputs a results.json file
    json_path = f"{val_dir}/results.json"
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                data = json.load(f)
            # Find the best epoch or just the only epoch in val
            if isinstance(data, list) and len(data) > 0:
                best = data[-1]
                map50 = best.get('metrics/mAP50(B)', 0)
                map5095 = best.get('metrics/mAP50-95(B)', 0)
                return map50, map5095
        except Exception:
            pass
            
    raise Exception("Validation failed to produce metrics")

def compare():
    import csv

    print("\n" + "="*60)
    print(f"{'Model':<12} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'OAK-D blob':>12}")
    print("-"*60)

    models_to_check = ['yolov5n', 'yolov6n', 'yolov8n', 'yolo26n']
    
    for name in models_to_check:
        pt = f"{PROJECT_DIR}/{name}/weights/best.pt"
        blob = "✓ yes" if name != 'yolo26n' else "✗ not yet"
        
        if not os.path.exists(pt):
            print(f"{name:<12} {'not trained yet':<26} {blob:>12}")
            continue
            
        csv_path_inside_weights = pt.replace('weights/best.pt', 'weights/results.csv')
        csv_path_parent = pt.replace('weights/best.pt', 'results.csv')
        # Check specific validation folder for v8 if parent is missing
        csv_path_val = f"{PROJECT_DIR}/{name}_val/results.csv" if name == 'yolov8n' else None
        
        csv_path = None
        if os.path.exists(csv_path_parent):
            csv_path = csv_path_parent
        elif os.path.exists(csv_path_inside_weights):
            csv_path = csv_path_inside_weights
        elif csv_path_val and os.path.exists(csv_path_val):
            csv_path = csv_path_val

        if not csv_path:
            # Fallback: run validation right now to get the numbers
            try:
                map50_val, map5095_val = run_val_if_missing(name, pt)
                map50 = f"{map50_val:.5f}"
                map5095 = f"{map5095_val:.5f}"
                print(f"{name:<12} {map50:>10} {map5095:>14} {blob:>12}")
            except Exception as e:
                import traceback
                print(f"{name:<12} {'validation failed':<26} {blob:>12}")
                traceback.print_exc()
            continue
            
        with open(csv_path) as f:
            rows = [{k.strip(): v.strip() for k,v in r.items()}
                    for r in csv.DictReader(f)]
        if not rows:
            continue
            
        # YOLOv5/v8 use different CSV headers sometimes
        try:
            best    = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', r.get('metrics/mAP_0.5', 0)) or 0))
            map50   = best.get('metrics/mAP50(B)',    best.get('metrics/mAP_0.5', 'N/A'))
            map5095 = best.get('metrics/mAP50-95(B)', best.get('metrics/mAP_0.5:0.95', 'N/A'))
            print(f"{name:<12} {map50:>10} {map5095:>14} {blob:>12}")
        except:
            print(f"{name:<12} {'error parsing CSV':<26} {blob:>12}")

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
        choices=['v5','v6','v8','v26','all', 'none'], default='all',
        help="Which model to train. Use 'none' to skip training and only run --export or --compare")
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
    # If args.train == 'none', do nothing here.

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