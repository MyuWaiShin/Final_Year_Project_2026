# Uni Computer Quick-Start Guide
# PDE3802 — Object Detection Pipeline

Run these steps **every time you log into the uni computer**.

---

## FIRST TIME ONLY (do once per machine)

### 1. Install Python 3.11

Open PowerShell and run:
```powershell
py install 3.11
```
Verify it worked:
```powershell
py -3.11 --version
# Should print: Python 3.11.x
```
> If `py` is not found, download Python 3.11.9 from:
> https://www.python.org/downloads/release/python-3119/

---

### 2. Fix PowerShell Execution Policy (one-time)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 3. Create the Virtual Environment (one-time)
```powershell
cd "c:\Users\ms3433\Downloads\Object Detection Pipeline\Object Detection Pipeline"

py -3.11 -m venv venv
```
---

### 4. Install Data Prep Packages (one-time)
```powershell
.\venv\Scripts\Activate.ps1

pip install -r Data_Preparation_V2\requirements_dataprep.txt
```

---

### 5. Install Annotation Packages (one-time, takes longer)

> **Follow this order exactly — skipping steps causes import errors.**

#### 5a. PyTorch with CUDA (RTX 3080 → CUDA 12.1)
```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 5b. GroundingDINO + missing deps + annotation packages
```powershell
pip install groundingdino-py
pip install scikit-learn roboflow
pip install -r Data_Preparation_V2\requirements_annotation.txt
```

> **Note:** Ignore warnings about `supervision==0.6.0` conflict — groundingdino-py works fine with newer supervision.

---

## EVERY SESSION (each time you log in)

```powershell
# 1. Navigate to project root
cd "c:\Users\ms3433\Downloads\Object Detection Pipeline\Object Detection Pipeline"

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Your prompt should now show (venv) at the start
```

That's it — the venv persists between logins, no reinstalling needed.

---

## RUN THE PIPELINE (in order)

```powershell
# Extract frames from raw videos
python Data_Preparation_V2\scripts\1_extract_frames.py

# Split into train / val / test  (75 / 15 / 15)
python Data_Preparation_V2\scripts\2_split_dataset.py

# Auto-annotate with GroundingDINO
python Data_Preparation_V2\scripts\3_auto_annotation.py

# Check dataset stats at any time
python Data_Preparation_V2\scripts\3_dataset_stats.py
```

---
# Visit this page for installation instructions: https://pytorch.org/get-started/locally/
---

## TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `py` not found | Run `py install 3.11` — if that fails, download from python.org |
| `Activate.ps1` not found | Make sure you're in the project ROOT, not inside the `venv` folder |
| `running scripts is disabled` | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip` not found | Activate the venv first (`.\venv\Scripts\Activate.ps1`) |
| GroundingDINO import error | Run `pip install groundingdino-py` before other annotation deps |
| `CUDA out of memory` | Close other GPU apps, or use CPU-only PyTorch |
