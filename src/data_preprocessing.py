import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------
# CONFIG
# -------------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

DATASETS = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

DROP_COLUMNS = [
    "Flow ID", "Timestamp",
    "Src IP", "Dst IP",
    "Source IP", "Destination IP",
    "Src Port", "Dst Port"
]

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess_dataset(file_name):
    print("\n" + "=" * 60)
    print(f"Processing dataset: {file_name}")

    path = os.path.join(RAW_DATA_DIR, file_name)
    df = pd.read_csv(path)

    original_samples = df.shape[0]

    # Strip spaces from column names (IMPORTANT FIX)
    df.columns = df.columns.str.strip()

    # Drop non-learning columns
    df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns],
            inplace=True, errors="ignore")

    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop missing values
    df.dropna(inplace=True)

    # -------- FIX: Detect label column safely --------
    if "Label" in df.columns:
        label_col = "Label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(f"Label column not found in {file_name}")

    # Split features and label
    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # Binary encoding
    y = y.apply(lambda x: 0 if x.strip().upper() == "BENIGN" else 1)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Original samples : {original_samples}")
    print(f"Clean samples    : {X_scaled.shape[0]}")
    print(f"Total features   : {X_scaled.shape[1]}")
    print(f"Benign samples   : {(y == 0).sum()}")
    print(f"Attack samples   : {(y == 1).sum()}")

    return X_scaled, y.values



# -------------------------------
# MAIN EXECUTION
# -------------------------------
summary = []

for idx, dataset in enumerate(DATASETS, start=1):
    X, y = preprocess_dataset(dataset)

    # Save processed dataset
    save_path = os.path.join(PROCESSED_DATA_DIR, f"day_{idx}_processed.npz")
    np.savez(save_path, X=X, y=y)

    summary.append({
        "Dataset": dataset,
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "Benign": int((y == 0).sum()),
        "Attack": int((y == 1).sum())
    })

# -------------------------------
# SAVE SUMMARY TABLE
# -------------------------------
summary_df = pd.DataFrame(summary)
summary_csv_path = os.path.join(PROCESSED_DATA_DIR, "dataset_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print("\n" + "=" * 60)
print("PHASE 1 COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nDATASET SUMMARY TABLE:\n")
print(summary_df)
print("\nSaved files:")
print(" - data/processed/day_X_processed.npz")
print(" - data/processed/dataset_summary.csv")
