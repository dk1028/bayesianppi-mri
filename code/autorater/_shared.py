from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CSV_ROOT = DATA_ROOT / "csv"
NIFTI_ROOT = DATA_ROOT / "nifti"
RESULTS_ROOT = REPO_ROOT / "results"
FIGS_ROOT = REPO_ROOT / "figs"
AUTORATER_ROOT = RESULTS_ROOT / "autorater"
AUTORATER_ROOT.mkdir(parents=True, exist_ok=True)
FIGS_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SHAPE = (64, 64, 64)
N_FOLDS = 5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_nifti_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return NIFTI_ROOT / path


def _safe_minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def brain_bbox(volume: np.ndarray, eps: float = 0.0) -> tuple[slice, slice, slice]:
    mask = np.isfinite(volume) & (volume > eps)
    if not mask.any():
        return slice(0, volume.shape[0]), slice(0, volume.shape[1]), slice(0, volume.shape[2])
    coords = np.array(np.where(mask))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))  # type: ignore[return-value]


def crop_or_pad_to_cube(volume: np.ndarray) -> np.ndarray:
    sx, sy, sz = brain_bbox(volume)
    cropped = volume[sx, sy, sz]
    target = int(max(cropped.shape))
    pads = []
    for dim in cropped.shape:
        total = target - dim
        before = total // 2
        after = total - before
        pads.append((before, after))
    return np.pad(cropped, pads, mode="constant", constant_values=0.0)


def preprocess_volume(nifti_path: Path) -> torch.Tensor:
    volume = nib.load(str(nifti_path)).get_fdata().astype(np.float32)
    volume = crop_or_pad_to_cube(volume)
    volume = _safe_minmax(volume)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    tensor = F.interpolate(
        tensor,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return tensor.squeeze(0)  # [1, 64, 64, 64]


class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.loc[idx]
        img = preprocess_volume(resolve_nifti_path(row["nifti_path"]))
        label = torch.tensor(float(row["H"]), dtype=torch.float32)
        return img, label


class Simple3DCNN(nn.Module):
    """Two-block lightweight 3D CNN matching the manuscript text."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(16 * 16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x


@dataclass
class TrainingArtifacts:
    model: Simple3DCNN
    train_losses: list[float]



def train_model(train_df: pd.DataFrame, seed: int = SEED) -> TrainingArtifacts:
    set_seed(seed)
    ds = MRIDataset(train_df)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = Simple3DCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses: list[float] = []
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        losses.append(running / max(len(dl), 1))
        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {losses[-1]:.4f}")
    return TrainingArtifacts(model=model, train_losses=losses)


@torch.no_grad()
def predict_df(model: nn.Module, df: pd.DataFrame) -> np.ndarray:
    model.eval()
    ds = MRIDataset(df)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds: list[float] = []
    for x, _ in dl:
        x = x.to(DEVICE)
        preds.extend(model(x).cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)


CANDIDATE_FULL_CSVS = [
    CSV_ROOT / "matched_cn_ad_labels_all.csv",
    CSV_ROOT / "matched_cn_ad_labels.csv",
]

CANDIDATE_6570_CSVS = [
    CSV_ROOT / "matched_cn_ad_labels_6570.csv",
    CSV_ROOT / "matched_cn_ad_labels_65_70.csv",
    CSV_ROOT / "matched_cn_ad_labels_age6570.csv",
]


def choose_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n - ".join(str(p) for p in paths)
    raise FileNotFoundError(f"Could not find any expected CSV. Checked:\n - {joined}")


REQUIRED_COLUMNS = {"label", "nifti_path"}
OPTIONAL_STRING_COLUMNS = ["subject_id", "Acq_Date", "Sex", "Age"]


def load_label_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["H"] = (df["label"].astype(str).str.upper() == "AD").astype(int)
    return df



def oof_predictions(df: pd.DataFrame, out_csv: Path, seed: int = SEED) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    oof = np.full(len(df), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["H"]), start=1):
        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        artifacts = train_model(train_df, seed=seed + fold)
        oof[valid_idx] = predict_df(artifacts.model, valid_df)

    result = df.copy()
    result["autorater_prediction"] = oof
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"Saved OOF predictions to: {out_csv}")
    return result



def single_holdout_predictions(df: pd.DataFrame, out_csv: Path, seed: int = SEED) -> pd.DataFrame:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["H"],
        random_state=seed,
    )
    artifacts = train_model(train_df.reset_index(drop=True), seed=seed)
    out = test_df.copy().reset_index(drop=True)
    out["autorater_prediction"] = predict_df(artifacts.model, out)
    out.to_csv(out_csv, index=False)
    print(f"Saved holdout predictions to: {out_csv}")
    return out



def load_mid_axial_slice(nifti_path: Path, processed: bool = False) -> np.ndarray:
    if processed:
        tensor = preprocess_volume(nifti_path).numpy()[0]
        sl = tensor[:, :, tensor.shape[2] // 2]
        return _safe_minmax(sl)
    volume = nib.load(str(nifti_path)).get_fdata().astype(np.float32)
    sl = volume[:, :, volume.shape[2] // 2]
    return _safe_minmax(sl)
