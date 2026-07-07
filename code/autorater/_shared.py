from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from fold_utils import make_subject_grouped_folds, validate_subject_fold_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CSV_ROOT = DATA_ROOT / "csv"
NIFTI_ROOT = DATA_ROOT / "nifti"
RESULTS_ROOT = REPO_ROOT / "results"
FIGS_ROOT = REPO_ROOT / "figs"
AUTORATER_ROOT = RESULTS_ROOT / "autorater"
AUTORATER_ROOT.mkdir(parents=True, exist_ok=True)
FIGS_ROOT.mkdir(parents=True, exist_ok=True)

CANDIDATE_FULL_CSVS = [
    CSV_ROOT / "matched_cn_ad_labels_all.csv",
    CSV_ROOT / "matched_cn_ad_labels.csv",
]
CANDIDATE_6570_CSVS = [
    CSV_ROOT / "matched_cn_ad_labels_6570.csv",
    CSV_ROOT / "matched_cn_ad_labels_65_70.csv",
]

def choose_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n - ".join(str(p) for p in paths)
    raise FileNotFoundError(f"Could not find any expected file:\n - {joined}")

SEED = 42
TARGET_SHAPE = (64, 64, 64)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_nifti_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else NIFTI_ROOT / path


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
        return (
            slice(0, volume.shape[0]),
            slice(0, volume.shape[1]),
            slice(0, volume.shape[2]),
        )
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
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return tensor.squeeze(0)


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
    """Two-block lightweight 3D CNN used in the manuscript pipeline."""

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
        return torch.sigmoid(self.fc2(x)).squeeze(1)


@dataclass
class TrainingArtifacts:
    model: Simple3DCNN
    train_losses: list[float]


def load_label_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required = {"subject_id", "nifti_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {sorted(missing)}")

    df["subject_id"] = df["subject_id"].astype(str).str.strip()

    if "H" in df.columns:
        df["H"] = pd.to_numeric(df["H"], errors="raise").astype(int)
    elif "label" in df.columns:
        labels = df["label"].astype(str).str.upper().str.strip()
        if not labels.isin(["AD", "CN"]).all():
            bad = sorted(labels[~labels.isin(["AD", "CN"])].unique())
            raise ValueError(f"Only AD/CN labels are supported; found {bad}")
        df["H"] = (labels == "AD").astype(int)
    else:
        raise ValueError("Input must contain H or label")

    return df.reset_index(drop=True)


def train_model(
    train_df: pd.DataFrame,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> TrainingArtifacts:
    set_seed(seed)
    ds = MRIDataset(train_df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = Simple3DCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        epoch_loss = running / max(len(dl), 1)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.4f}")

    return TrainingArtifacts(model=model, train_losses=losses)


@torch.no_grad()
def predict_df(
    model: nn.Module,
    df: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    ds = MRIDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds: list[float] = []
    for x, _ in dl:
        x = x.to(device)
        preds.extend(model(x).detach().cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)


def oof_predictions(
    df: pd.DataFrame,
    out_csv: Path,
    *,
    seed: int = 42,
    n_folds: int = 5,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: torch.device | None = None,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    device = auto_device() if device is None else device
    print("Device:", device)

    oof = np.full(len(df), np.nan, dtype=float)
    fold_assignment = np.full(len(df), -1, dtype=int)

    folds = make_subject_grouped_folds(
        df,
        subject_col="subject_id",
        label_col="H",
        n_splits=n_folds,
        seed=seed,
    )

    fold_logs: list[dict[str, object]] = []

    for split in folds:
        print(f"\n=== Fold {split.fold}/{n_folds} ===")
        train_df = df.iloc[split.train_idx].reset_index(drop=True)
        valid_df = df.iloc[split.valid_idx].reset_index(drop=True)

        artifacts = train_model(
            train_df,
            seed=seed + split.fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        oof[split.valid_idx] = predict_df(
            artifacts.model,
            valid_df,
            batch_size=batch_size,
            device=device,
        )
        fold_assignment[split.valid_idx] = split.fold

        fold_logs.append(
            {
                "fold": split.fold,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "train_subjects": int(train_df["subject_id"].nunique()),
                "valid_subjects": int(valid_df["subject_id"].nunique()),
                "losses": artifacts.train_losses,
            }
        )

        del artifacts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof).any() or (fold_assignment < 1).any():
        raise RuntimeError("Some rows did not receive OOF predictions or folds")

    result = df.copy()
    result["autorater_prediction"] = oof
    result["A_prob"] = oof
    result["A_class_0p5"] = (oof >= 0.5).astype(int)
    result["oof_fold"] = fold_assignment

    fold_qc = validate_subject_fold_assignment(result)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    metadata = {
        "rows": len(result),
        "subjects": int(result["subject_id"].nunique()),
        "folds": n_folds,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        **fold_qc,
        "fold_logs": fold_logs,
    }
    out_csv.with_suffix(".run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Saved:", out_csv)
    print(json.dumps(fold_qc, indent=2))
    return result


def single_holdout_predictions(
    df: pd.DataFrame,
    out_csv: Path,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Optional grouped holdout; not used for manuscript OOF analyses."""
    device = auto_device() if device is None else device
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(df, y=df["H"], groups=df["subject_id"])
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
    if overlap:
        raise RuntimeError("Subject overlap in grouped holdout")

    artifacts = train_model(
        train_df,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    out = test_df.copy()
    out["autorater_prediction"] = predict_df(
        artifacts.model,
        out,
        batch_size=batch_size,
        device=device,
    )
    out.to_csv(out_csv, index=False)
    return out


def load_mid_axial_slice(nifti_path: Path, processed: bool = False) -> np.ndarray:
    if processed:
        tensor = preprocess_volume(nifti_path).numpy()[0]
        sl = tensor[:, :, tensor.shape[2] // 2]
        return _safe_minmax(sl)
    volume = nib.load(str(nifti_path)).get_fdata().astype(np.float32)
    sl = volume[:, :, volume.shape[2] // 2]
    return _safe_minmax(sl)
