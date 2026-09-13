"""Step 1: load UCI HAR raw inertial signals into (N, 128, 6) windows.

Course concepts applied here:
  * windowing: 128 samples @ 50 Hz = 2.56 s, 50 % overlap (as recorded by the dataset)
  * grouped evaluation split: train / val / test never share a subject
  * train/deploy parity: the per-channel mean/std computed here are exported to the
    firmware so the ESP32 applies the identical preprocessing.
"""
import os, numpy as np
from common import *

def load_split(split):
    X = np.stack([np.loadtxt(os.path.join(DATA_DIR, split, "Inertial Signals", f"{s}_{split}.txt"))
                  for s in SIGNALS], axis=-1).astype(np.float32)          # (N,128,6)
    y = np.loadtxt(os.path.join(DATA_DIR, split, f"y_{split}.txt")).astype(np.int64) - 1
    subj = np.loadtxt(os.path.join(DATA_DIR, split, f"subject_{split}.txt")).astype(np.int64)
    return X, y, subj

if __name__ == "__main__":
    Xtr, ytr, str_ = load_split("train")
    Xte, yte, ste = load_split("test")
    val_mask = np.isin(str_, VAL_SUBJECTS)
    Xva, yva, sva = Xtr[val_mask], ytr[val_mask], str_[val_mask]
    Xtr, ytr, str_ = Xtr[~val_mask], ytr[~val_mask], str_[~val_mask]

    mean = Xtr.reshape(-1, N_CH).mean(0)
    std = Xtr.reshape(-1, N_CH).std(0)
    assert not set(str_) & set(sva) and not set(str_) & set(ste) and not set(sva) & set(ste)

    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, "dataset.npz"),
        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
        subj_tr=str_, subj_va=sva, subj_te=ste, mean=mean, std=std)
    info = {"train_windows": int(len(ytr)), "val_windows": int(len(yva)), "test_windows": int(len(yte)),
            "train_subjects": sorted(map(int, set(str_))), "val_subjects": sorted(map(int, set(sva))),
            "test_subjects": sorted(map(int, set(ste))),
            "window": WINDOW, "hop": HOP, "fs_hz": FS_HZ, "channels": SIGNALS,
            "mean": mean.tolist(), "std": std.tolist(),
            "class_counts_train": np.bincount(ytr).tolist()}
    save_json(info, "dataset_info.json")
    print(json.dumps(info, indent=2))
