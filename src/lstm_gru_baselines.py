import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
PLOT_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 256

# -------------------------------
# LOAD DATA
# -------------------------------
X = np.load(os.path.join(DATA_DIR, "X_sequences.npy"))
y = np.load(os.path.join(DATA_DIR, "y_sequences.npy"))

print(f"Loaded data shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# METRICS FUNCTION
# -------------------------------
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} PERFORMANCE")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return acc, prec, rec, f1

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------------
# LSTM MODEL
# -------------------------------
lstm_model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining LSTM model...")
lstm_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

y_pred_lstm = (lstm_model.predict(X_test) > 0.5).astype(int).ravel()
lstm_metrics = evaluate_model(y_test, y_pred_lstm, "LSTM")

plot_confusion(
    y_test, y_pred_lstm,
    "LSTM Confusion Matrix",
    os.path.join(PLOT_DIR, "confusion_lstm.png")
)

# -------------------------------
# GRU MODEL
# -------------------------------
gru_model = Sequential([
    GRU(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1, activation="sigmoid")
])

gru_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining GRU model...")
gru_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

gru_model.save(os.path.join(MODEL_DIR, "gru_model.h5"))

y_pred_gru = (gru_model.predict(X_test) > 0.5).astype(int).ravel()
gru_metrics = evaluate_model(y_test, y_pred_gru, "GRU")

plot_confusion(
    y_test, y_pred_gru,
    "GRU Confusion Matrix",
    os.path.join(PLOT_DIR, "confusion_gru.png")
)

print("\nPHASE 3 COMPLETED SUCCESSFULLY")
