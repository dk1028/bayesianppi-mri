import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- paths relative to repo root ---
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATA_ROOT    = REPO_ROOT / "data"
CSV_ROOT     = DATA_ROOT / "csv"
NIFTI_ROOT   = DATA_ROOT / "nifti"
RESULTS_ROOT = REPO_ROOT / "results"
AUTORATER_ROOT = RESULTS_ROOT / "autorater"
AUTORATER_ROOT.mkdir(parents=True, exist_ok=True)

# 1) Settings
CSV_PATH    = CSV_ROOT / "matched_cn_ad_labels_6570.csv"   # 65–70 subset CSV
BATCH_SIZE  = 2
EPOCHS      = 5
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2) Dataset definition
class MRIDataset(Dataset):
    def __init__(self, df, nifti_root: Path):
        self.df = df.reset_index(drop=True)
        self.nifti_root = nifti_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # --- resolve NIfTI path ---
        # row['nifti_path'] 가 절대경로면 그대로, 아니면 data/nifti 아래에 붙이기
        nifti_path = Path(row["nifti_path"])
        if not nifti_path.is_absolute():
            nifti_path = self.nifti_root / nifti_path

        img = nib.load(str(nifti_path)).get_fdata().astype("float32")  # [D, H, W]
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

        # Numeric label H: 0 for CN, 1 for AD
        label = torch.tensor(row["H"], dtype=torch.float32)

        return img, label


# 3) Simple 3D CNN
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.pool  = nn.MaxPool3d(2)
        self.fc1   = nn.Linear(16 * 16 * 16 * 16, 64)
        self.fc2   = nn.Linear(64, 1)

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

    # label == "AD" → 1, label == "CN" → 0 (이미 H가 있으면 덮어씌워도 무방)
    df["H"] = (df["label"] == "AD").astype(int)

    # Train / test split stratified on H
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["H"],
        random_state=42,
    )

    train_ds = MRIDataset(train_df, NIFTI_ROOT)
    test_ds  = MRIDataset(test_df,  NIFTI_ROOT)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 5) Model, loss, optimizer
    model     = Simple3DCNN().to(DEVICE)
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

    # 7) Predict on test set and save
    model.eval()
    test_preds  = []
    test_labels = []

    with torch.no_grad():
        for X, y in test_dl:
            X = X.to(DEVICE)
            out = model(X).cpu().numpy().tolist()
            test_preds.extend(out)
            test_labels.extend(y.tolist())

    out_df = test_df.copy().reset_index(drop=True)
    out_df["autorater_prediction"] = test_preds
    out_df["H"] = out_df["H"].astype(int)

    test_out_path = AUTORATER_ROOT / "autorater_predictions_6570_test.csv"
    out_df.to_csv(test_out_path, index=False)
    print(f"✅ {test_out_path.name} generated at {test_out_path}")

    # 8) Predict on all data and save
    all_ds = MRIDataset(df, NIFTI_ROOT)
    all_dl = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X, _ in all_dl:
            X = X.to(DEVICE)
            out = model(X).cpu().numpy().tolist()
            all_preds.extend(out)

    df_out = df.copy().reset_index(drop=True)
    df_out["autorater_prediction"] = all_preds
    df_out["H"] = df_out["H"].astype(int)

    all_out_path = AUTORATER_ROOT / "autorater_predictions_6570_all.csv"
    df_out.to_csv(all_out_path, index=False)
    print(f"✅ {all_out_path.name} generated at {all_out_path}")


if __name__ == "__main__":
    main()
