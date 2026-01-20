import os
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

WINDOW_SIZE = 10
MAX_SEQ_PER_DAY = 50000  # MEMORY-SAFE LIMIT
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# -------------------------------
# SEQUENCE CREATION FUNCTION
# -------------------------------
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq)

# -------------------------------
# MAIN EXECUTION
# -------------------------------
X_all_seq = []
y_all_seq = []

print("\n" + "=" * 60)
print("PHASE 2: SEQUENCE CREATION (MEMORY SAFE)")
print("=" * 60)

for i in range(1, 9):
    file_path = os.path.join(PROCESSED_DATA_DIR, f"day_{i}_processed.npz")
    data = np.load(file_path)

    X = data["X"]
    y = data["y"]

    X_seq, y_seq = create_sequences(X, y, WINDOW_SIZE)

    print(f"\nDay {i}:")
    print(f"  Total sequences before sampling: {X_seq.shape[0]}")

    # -------- SAMPLING STEP --------
    if X_seq.shape[0] > MAX_SEQ_PER_DAY:
        indices = np.random.choice(X_seq.shape[0], MAX_SEQ_PER_DAY, replace=False)
        X_seq = X_seq[indices]
        y_seq = y_seq[indices]

    print(f"  Sequences after sampling: {X_seq.shape[0]}")
    print(f"  Sequence shape: {X_seq.shape}")

    X_all_seq.append(X_seq)
    y_all_seq.append(y_seq)

# Merge sampled sequences
X_final = np.concatenate(X_all_seq, axis=0)
y_final = np.concatenate(y_all_seq, axis=0)

# Save final dataset
np.save(os.path.join(OUTPUT_DIR, "X_sequences.npy"), X_final)
np.save(os.path.join(OUTPUT_DIR, "y_sequences.npy"), y_final)

print("\n" + "=" * 60)
print("PHASE 2 COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Final sequence dataset shape: {X_final.shape}")
