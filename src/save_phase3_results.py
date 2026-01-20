import os
import pandas as pd

# Create directory
os.makedirs("results/tables", exist_ok=True)

# Phase 3 baseline results
data = {
    "Model": ["LSTM", "GRU"],
    "Accuracy": [0.9884, 0.9882],
    "Precision": [0.9839, 0.9842],
    "Recall": [0.9556, 0.9541],
    "F1-score": [0.9696, 0.9689]
}

df = pd.DataFrame(data)

# Save CSV
output_path = "results/tables/phase3_baseline_results.csv"
df.to_csv(output_path, index=False)

print("PHASE 3 RESULT TABLE SAVED SUCCESSFULLY")
print(df)
