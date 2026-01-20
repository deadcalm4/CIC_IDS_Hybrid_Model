import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
PLOT_DIR = "results/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 256

# -------------------------------
# LOAD FEATURES (FROM PHASE 4)
# -------------------------------
X = np.load(os.path.join(DATA_DIR, "X_sequences.npy"))
y = np.load(os.path.join(DATA_DIR, "y_sequences.npy"))

from tensorflow.keras.models import load_model
lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
gru_model = load_model(os.path.join(MODEL_DIR, "gru_model.h5"))

lstm_feat = lstm_model.layers[0](X).numpy()
gru_feat = gru_model.layers[0](X).numpy()

# -------------------------------
# AUTOENCODER FUNCTION
# -------------------------------
def build_autoencoder(input_dim):
    inp = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(inp)
    decoded = Dense(input_dim, activation="linear")(encoded)
    autoencoder = Model(inp, decoded)
    encoder = Model(inp, encoded)
    autoencoder.compile(optimizer=Adam(), loss="mse")
    return autoencoder, encoder

# -------------------------------
# TRAIN SAE (LSTM FEATURES)
# -------------------------------
print("\nTraining SAE on LSTM features...")
sae_lstm, encoder_lstm = build_autoencoder(lstm_feat.shape[1])
history_lstm = sae_lstm.fit(
    lstm_feat, lstm_feat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)

encoder_lstm.save(os.path.join(MODEL_DIR, "encoder_lstm.h5"))

# -------------------------------
# TRAIN SAE (GRU FEATURES)
# -------------------------------
print("\nTraining SAE on GRU features...")
sae_gru, encoder_gru = build_autoencoder(gru_feat.shape[1])
history_gru = sae_gru.fit(
    gru_feat, gru_feat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)

encoder_gru.save(os.path.join(MODEL_DIR, "encoder_gru.h5"))

# -------------------------------
# PLOT LOSS CURVES
# -------------------------------
plt.figure()
plt.plot(history_lstm.history["loss"], label="LSTM SAE Loss")
plt.plot(history_gru.history["loss"], label="GRU SAE Loss")
plt.title("SAE Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "sae_training_loss.png"))
plt.close()

print("\nPHASE 5 COMPLETED SUCCESSFULLY")
