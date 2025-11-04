# fNIRS Torch Starter (Synthetic Data)

Learn **barebones PyTorch** with **synthetic fNIRS** signals. You will:
1) Generate simple SportX-like synthetic fNIRS HbO/HbR signals
2) Window the time series into examples
3) Train a tiny 1D CNN to classify **Task** vs **Rest**

> Keep it simple: 10 Hz sampling, block design, canonical HRF + noise + a few motion blips.

---

## Quickstart

### 0) Make a new repo
```bash
# create a GitHub repo named fnirs-torch-starter first (on github.com)
# then on your laptop:
git clone https://github.com/xveqc3204/fnirs-torch-starter.git
cd fnirs-torch-starter
```

### 1) Create & activate a Python environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**What this does**
- `python -m venv .venv` makes an isolated Python.
- `activate` uses that Python so your packages stay clean per-project.

### 2) Install deps
```bash
pip install -r requirements.txt
```

**What this does**
- Installs PyTorch, NumPy, SciPy, Matplotlib, scikit-learn, and MNE.

### 3) Generate data
```bash
python src/generate_data.py --out data/sportx_synth.npz
```
**What this does**
- Creates a synthetic multi‑channel time series with a block task design,
- adds HRF, cardiac/resp/Mayer waves, white noise, and a couple motion blips,
- windows the signal into (C × T) examples and labels,
- saves to `data/sportx_synth.npz`.

### 4) Train the tiny CNN
```bash
python src/train.py --data data/sportx_synth.npz --epochs 10
```
**What this does**
- Loads the windowed tensors,
- runs a minimal 1D‑CNN on CPU,
- prints loss/accuracy.

### 5) (Optional) Visualize a sample
```bash
python src/visualize_sample.py --data data/sportx_synth.npz
```

---

## Project layout
```
fnirs-torch-starter
├── data/                       # .npz gets written here
├── notebooks/                  # space for your own experiments
├── src/
│   ├── generate_data.py        # build synthetic fNIRS + window + save
│   ├── dataset.py              # PyTorch Dataset for the .npz
│   ├── model.py                # tiny 1D‑CNN
│   ├── train.py                # training loop
│   └── visualize_sample.py     # quick plot of one window
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Learning notes (super short)
- **fNIRS basics:** We mimic HbO/HbR changes by convolving a block‑design “neural” train with a **canonical HRF** (difference of gammas).
- **Noise:** add cardiac (~1.2 Hz), respiration (~0.25 Hz), Mayer waves (~0.1 Hz), plus small white noise.
- **Artifacts:** inject an occasional spike or baseline shift to keep it real.
- **Why 1D‑CNN?** It learns the **shape** of the HRF within windows without manual features.
