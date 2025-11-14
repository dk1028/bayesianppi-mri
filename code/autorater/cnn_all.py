# cnn_all.py — 3D CNN autorater for full cohort

import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# --- paths relative to repo root ---
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATA_ROOT    = REPO_ROOT / "data"
CSV_ROOT     = DATA_ROOT / "csv"
NIFTI_ROOT   = DATA_ROOT / "nifti"
RESULTS_ROOT = REPO_ROOT / "results"
AUTORATER_ROOT = RESULTS_ROOT / "autorater"
AUTORATER_ROOT.mkdir(parents=True, exist_ok=True)

# --- reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1) Settings
CSV_PATH   = CSV_ROOT / "matched_cn_ad_labels_all.csv"   # full cohort
BATCH_SIZE = 2
EPOCHS     = 5
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2) Dataset definition
class MRIDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # Resolve NIfTI path: allow either absolute or relative to NIFTI_ROOT
        nifti_path = Path(row["nifti_path"])
        if not nifti_path.is_absolute():
            nifti_path = NIFTI_ROOT / nifti_path

        # Load NIfTI and convert: numpy → tensor
        img = nib.load(str(nifti_path)).get_fdata().astype("float32")
        img = torch.from_numpy(img).unsqueeze(0)  # [1, D, H, W]

        # Resize to 64 x 64 x 64
        img = F.interpolate(
            img.unsqueeze(0),  # [1, 1, D, H, W]
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # [1, 64, 64, 64]

        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-5)

        # Use numeric label H (0 for CN, 1 for AD)
        label = torch.tensor(row["H"], dtype=torch.float32)

        return img, label


# 3) Simple 3D CNN model
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(16 * 16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 8, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 16, 16, 16, 16]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x


def main():
    # 4) Load data, create H column, and split
    df = pd.read_csv(CSV_PATH)

    # Create H column: CN → 0, AD → 1
    df["H"] = (df["label"] == "AD").astype(int)

    # Train / test split using H for stratification
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["H"], random_state=SEED
    )

    train_ds = MRIDataset(train_df)
    test_ds = MRIDataset(test_df)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5) Model, loss, optimizer
    model = Simple3DCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 6) Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch}/{EPOCHS}  Loss: {avg_loss:.4f}")

    # 7) Predict on test set and save (with H column)
    model.eval()
    test_preds = []

    with torch.no_grad():
        for X, _ in test_dl:
            X = X.to(DEVICE)
            out = model(X).cpu().numpy().tolist()
            test_preds.extend(out)

    out_df = test_df.copy().reset_index(drop=True)
    out_df["autorater_prediction"] = test_preds

    test_out_path = AUTORATER_ROOT / "autorater_predictions_test.csv"
    out_df.to_csv(test_out_path, index=False)
    print(f"✅ Test predictions saved to: {test_out_path}")

    # 8) Predict on all data and save (with H column)
    all_ds = MRIDataset(df)
    all_dl = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X, _ in all_dl:
            X = X.to(DEVICE)
            out = model(X).cpu().numpy().tolist()
            all_preds.extend(out)

    df_out = df.copy().reset_index(drop=True)
    df_out["autorater_prediction"] = all_preds

    all_out_path = AUTORATER_ROOT / "autorater_predictions_all.csv"
    df_out.to_csv(all_out_path, index=False)
    print(f"✅ All predictions saved to: {all_out_path}")


if __name__ == "__main__":
    main()
