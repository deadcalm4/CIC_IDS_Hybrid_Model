import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# LOAD PHASE-WISE RESULTS
# -------------------------------
phase3 = pd.read_csv("results/tables/phase3_baseline_results.csv")
phase4 = pd.read_csv("results/tables/phase4_dl_svm_results.csv")
phase6 = pd.read_csv("results/tables/phase6_hybrid_results.csv")

# -------------------------------
# COMBINE RESULTS
# -------------------------------
final_df = pd.concat([phase3, phase4, phase6], ignore_index=True)

# Save final table
os.makedirs("results/tables", exist_ok=True)
final_df.to_csv("results/tables/final_comparison_results.csv", index=False)

print("\nFINAL COMPARISON TABLE")
print(final_df)

# -------------------------------
# PLOTTING
# -------------------------------
os.makedirs("results/plots", exist_ok=True)

# Accuracy plot
plt.figure()
plt.bar(final_df["Model"], final_df["Accuracy"])
plt.xticks(rotation=45, ha="right")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("results/plots/final_accuracy_comparison.png")
plt.close()

# F1-score plot
plt.figure()
plt.bar(final_df["Model"], final_df["F1-score"])
plt.xticks(rotation=45, ha="right")
plt.title("Model F1-score Comparison")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig("results/plots/final_f1_comparison.png")
plt.close()

print("\nPHASE 7 COMPLETED SUCCESSFULLY")
