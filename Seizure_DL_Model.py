import os
import re
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

def load_mea_dataset(data_dir):
    """
    Loads MEA seizure/non-seizure data from .mat files with bandpass and comb-filtered versions.
    Each sample = [time x channels x 4 bands]
    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    print(f"📂 Found {len(files)} .mat files")


    # Group by patient + event (ignoring band info)
    groups = {}
    for f in files:
        base = re.sub(r"(000_030|030_080|080_250|Comb)\.mat$", "", f)
        groups.setdefault(base, []).append(f)
    
    print(f"✅ Detected {len(groups)} patient-event groups\n")


    for g, fl in groups.items():
        print(f"{g}: {[match.group(1) for f in fl if (match := re.search(r'(000_030|030_080|080_250|Comb)', f))]}")


    X, y, meta = [], [], []
    skipped_incomplete = 0

    for gname, flist in tqdm(groups.items(), desc="📦 Building dataset"):
        bands = {}
        for f in flist:
            match = re.search(r'(000_030|030_080|080_250|Comb)', f)
            if match:
                bands[match.group(1).lower()] = os.path.join(data_dir, f)
            #print(bands)

        # Ensure all 4 bands exist
        if len(bands) != 4:
            skipped_incomplete += 1
            continue

        # Load all 4 bands
        arrays = []
        for bname in ['000_030', '030_080', '080_250', 'comb']:
            mat = loadmat(bands[bname])
            data = mat.get("data")
            #print(f, data.shape)
            #print(data)
            if data is None:
                skipped_incomplete += 1
                break
            if data.ndim == 1:
                data = data[np.newaxis, :]  # ensure 2D: time × channels
            arrays.append(data)
        else:
            # Truncate all to shortest common time length
            min_len = min(a.shape[0] for a in arrays)
            arrays = [a[:min_len, :] for a in arrays]

            # Stack along last axis (channels × time × 4 bands)
            stacked = np.stack(arrays, axis=-1)
            X.append(stacked)
            y.append(0 if "NonSz" in gname else 1)
            meta.append(gname)


    # --- ✅ Fix broadcasting issue here ---
    X_obj = np.empty(len(X), dtype=object)
    for i, arr in enumerate(X):
        X_obj[i] = arr
    # --------------------------------------

    print(f"\n✅ Final dataset summary:")
    print(f"   X samples: {len(X_obj)}")
    print(f"   Example shape: {X_obj[0].shape} (time × channels × 4 bands)")
    print(f"   Skipped incomplete: {skipped_incomplete}")
    print(f"   Channel counts: {[x.shape[1] for x in X_obj]}")

    return X_obj, np.array(y), np.array(meta)
    


# Example usage:
data_dir = "/Users/asifsalekin/Mebin_ML/Bradley_Greger_Project/Seizure_data/train_combined"
X, y, meta = load_mea_dataset(data_dir)

## Deep Learning Model Building

# Imports

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# -------------------------
# USER CONFIG
# -------------------------
FS = 500                   # sampling frequency (Hz). Change if your data uses different FS
WINDOW_SEC = 2             # window length in seconds
OVERLAP = 0.5              # fraction overlap (0.0-0.9)
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
RANDOM_SEED = 42
NUM_WORKERS = 0           # dataloader workers. Could use 4, but Jupyterlab is not allowing that
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# HELPERS: patient id extraction
# -------------------------
def get_patient_id(event_name: str) -> str:
    """
    Extract patient id by splitting at 'Sz' or 'NonSz' or 'Base' tokens.
    Adjust if your event naming scheme differs.
    """
    # Try split by common tokens
    m = re.split(r'(Sz|NonSz|Base|Non|_Sz|_NonSz)', event_name)
    if len(m) > 0:
        return m[0]
    return event_name


# -------------------------
# 1) SEGMENT + PAD + PREPARE SAMPLES
# -------------------------
def prepare_windows_from_X(X_obj, y, meta,
                           fs=FS, window_sec=WINDOW_SEC, overlap=OVERLAP):
    """
    Input:
      X_obj: object array/list of arrays, each array shape = (time, channels, bands)
      y: labels aligned with X_obj (list/array)
      meta: meta names aligned
    Returns:
      windows: np.array shape (N_windows, bands, channels, window_size)
      labels: np.array (N_windows,)
      meta_windows: list of (event_name, start_idx) tuples
    """
    window_size = int(window_sec * fs)
    step = int(window_size * (1 - overlap))
    assert step >= 1

    # find max channels across all events to pad channels dimension
    channel_counts = [arr.shape[1] for arr in X_obj]
    max_channels = max(channel_counts)

    windows = []
    labels = []
    meta_windows = []

    for arr, lbl, mname in zip(X_obj, y, meta):
        # arr shape assumed (time, channels, bands)
        timepoints, channels, bands = arr.shape
        # truncate if time < window_size
        if timepoints < window_size:
            # skip too-short events (alternatively: pad in time; but skip here)
            continue
        # sliding windows
        for start in range(0, timepoints - window_size + 1, step):
            w = arr[start:start + window_size, :, :]  # (window_size, channels, bands)
            # transpose to (bands, channels, time)
            w = np.transpose(w, (2, 1, 0)).astype(np.float32)
            # pad channels to max_channels: shape -> (bands, max_channels, time)
            if channels < max_channels:
                pad = np.zeros((bands, max_channels - channels, window_size), dtype=np.float32)
                w = np.concatenate([w, pad], axis=1)
            windows.append(w)
            labels.append(lbl)
            meta_windows.append((mname, start))
    if len(windows) == 0:
        raise ValueError("No windows produced — lower WINDOW_SEC or adjust FS/overlap.")
    X_windows = np.stack(windows, axis=0)   # (N, bands, max_channels, window_size)
    y_windows = np.array(labels, dtype=np.int64)
    return X_windows, y_windows, meta_windows, max_channels


# -------------------------
# 2) Data normalization (per-band global z-score)
# -------------------------
def normalize_windows_global(X_windows):
    # X_windows shape (N, bands, channels, time)
    # compute mean/std per band across all samples/channels/time
    mean = X_windows.mean(axis=(0,2,3), keepdims=True)  # shape (1,bands,1,1)
    std = X_windows.std(axis=(0,2,3), keepdims=True) + 1e-8
    X_norm = (X_windows - mean) / std
    return X_norm, mean, std


# -------------------------
# 3) PyTorch Dataset
# -------------------------
class MEAWindowDataset(Dataset):
    def __init__(self, X, y):
        # X shape (N, bands, channels, time)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # for BCEWithLogitsLoss use float
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# 4) Simple 2D-CNN model
#    Input: (batch, bands, channels, time)
# -------------------------
class SimpleSeizureCNN(nn.Module):
    def __init__(self, in_bands=4, in_ch=None, in_time=None):
        super().__init__()
        # a few conv layers; keep flexible for varying channel/time sizes
        self.conv1 = nn.Conv2d(in_channels=in_bands, out_channels=32, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        # Global pooling -> classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 1)  # binary output (logit)
    def forward(self, x):
        # x: (B, bands, channels, time)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.global_pool(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.fc(x)             # (B, 1)
        return x.squeeze(1)        # (B,)


# -------------------------
# 5) Training / validation function
# -------------------------
def train_model(X_windows, y_windows, meta_windows,
                patient_wise_split=True, test_size=0.2,
                batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR):
    # Option: patient-wise split to avoid leakage
    if patient_wise_split:
        # build patient list from meta_windows: meta_windows contains (event_name, start)
        patient_ids = [get_patient_id(m[0]) for m in meta_windows]
        unique_patients = list(set(patient_ids))
        # split patients
        train_p, val_p = train_test_split(unique_patients, test_size=test_size, random_state=RANDOM_SEED)
        # mask windows
        train_idx = [i for i,p in enumerate(patient_ids) if p in train_p]
        val_idx = [i for i,p in enumerate(patient_ids) if p in val_p]
    else:
        idx = np.arange(len(y_windows))
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=RANDOM_SEED, stratify=y_windows)

    X_train = X_windows[train_idx]
    y_train = y_windows[train_idx]
    X_val = X_windows[val_idx]
    y_val = y_windows[val_idx]

    # normalize using train stats only
    X_train, mean, std = normalize_windows_global(X_train)
    # apply same normalization to val
    X_val = (X_val - mean) / std

    # datasets + loaders
    train_ds = MEAWindowDataset(X_train, y_train)
    val_ds = MEAWindowDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # model
    in_bands = X_train.shape[1]
    model = SimpleSeizureCNN(in_bands=in_bands).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # eval
        model.eval()
        preds, trues = [], []
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            with torch.no_grad():
                logits = model(xb).cpu().numpy()
            preds.extend(logits.tolist())
            trues.extend(yb.numpy().tolist())

        # compute metrics (convert logits to probs using sigmoid)
        probs = 1.0 / (1.0 + np.exp(-np.array(preds)))
        pred_labels = (probs >= 0.5).astype(int)
        acc = accuracy_score(trues, pred_labels)
        try:
            auc = roc_auc_score(trues, probs)
        except Exception:
            auc = float('nan')

        print(f"Epoch {epoch}/{epochs}  TrainLoss={np.mean(train_losses):.4f}  ValAcc={acc:.4f}  ValAUC={auc:.4f}")

        if not np.isnan(auc) and auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), "best_seizure_cnn.pt")
    print("Training complete. Best Val AUC:", best_val_auc)
    return model



## Run model

X_windows, y_windows, meta_windows, max_ch = prepare_windows_from_X(X, y, meta)
model = train_model(X_windows, y_windows, meta_windows)