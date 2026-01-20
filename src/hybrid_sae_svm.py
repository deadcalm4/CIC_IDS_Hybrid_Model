import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import load_model

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
PLOT_DIR = "results/plots"
TABLE_DIR = "results/tables"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
X = np.load(os.path.join(DATA_DIR, "X_sequences.npy"))
y = np.load(os.path.join(DATA_DIR, "y_sequences.npy"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# LOAD MODELS
# -------------------------------
lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
gru_model = load_model(os.path.join(MODEL_DIR, "gru_model.h5"))

encoder_lstm = load_model(os.path.join(MODEL_DIR, "encoder_lstm.h5"))
encoder_gru = load_model(os.path.join(MODEL_DIR, "encoder_gru.h5"))

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
print("Extracting LSTM features...")
X_train_lstm = lstm_model.layers[0](X_train).numpy()
X_test_lstm = lstm_model.layers[0](X_test).numpy()

print("Extracting GRU features...")
X_train_gru = gru_model.layers[0](X_train).numpy()
X_test_gru = gru_model.layers[0](X_test).numpy()

# -------------------------------
# SAE COMPRESSION
# -------------------------------
print("Compressing LSTM features...")
X_train_lstm_sae = encoder_lstm.predict(X_train_lstm)
X_test_lstm_sae = encoder_lstm.predict(X_test_lstm)

print("Compressing GRU features...")
X_train_gru_sae = encoder_gru.predict(X_train_gru)
X_test_gru_sae = encoder_gru.predict(X_test_gru)

# -------------------------------
# SVM TRAINING FUNCTION
# -------------------------------
def train_eval_svm(X_tr, X_te, y_tr, y_te, name):
    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(X_tr, y_tr)

    y_pred = svm.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)

    print(f"\n{name} PERFORMANCE")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y_te, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"confusion_{name.lower().replace(' ', '_')}.png"))
    plt.close()

    joblib.dump(svm, os.path.join(MODEL_DIR, f"svm_{name.lower().replace(' ', '_')}.pkl"))

    return acc, prec, rec, f1

# -------------------------------
# TRAIN FINAL HYBRID MODELS
# -------------------------------
lstm_hybrid = train_eval_svm(
    X_train_lstm_sae, X_test_lstm_sae, y_train, y_test, "LSTM_SAE_SVM"
)

gru_hybrid = train_eval_svm(
    X_train_gru_sae, X_test_gru_sae, y_train, y_test, "GRU_SAE_SVM"
)

# -------------------------------
# SAVE RESULT TABLE
# -------------------------------
df = pd.DataFrame({
    "Model": ["LSTM + SAE + SVM", "GRU + SAE + SVM"],
    "Accuracy": [lstm_hybrid[0], gru_hybrid[0]],
    "Precision": [lstm_hybrid[1], gru_hybrid[1]],
    "Recall": [lstm_hybrid[2], gru_hybrid[2]],
    "F1-score": [lstm_hybrid[3], gru_hybrid[3]]
})

df.to_csv(os.path.join(TABLE_DIR, "phase6_hybrid_results.csv"), index=False)

print("\nPHASE 6 COMPLETED SUCCESSFULLY")
print(df)
