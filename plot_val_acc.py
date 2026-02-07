# Plot the validation accuracy using the summary logs from training

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "checkpoints/fastvit-proj/summary.csv"

df = pd.read_csv(CSV_PATH)

# Convert safely (forces bad rows to NaN)
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df["eval_top1"] = pd.to_numeric(df["eval_top1"], errors="coerce")

# Drop garbage rows
df = df.dropna(subset=["epoch", "eval_top1"])

if len(df) == 0:
    raise RuntimeError("No valid validation data found in summary.csv")

acc = df["eval_top1"]

best_idx = acc.idxmax()
best_epoch = int(df.loc[best_idx, "epoch"])
best_acc = acc.max()

plt.figure(figsize=(10,6))
plt.plot(df["epoch"], acc, marker="o", linewidth=2)

plt.axhline(best_acc, linestyle="--")
plt.title(f"Validation Top-1 (Best={best_acc:.2f}% @ epoch {best_epoch})")

plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy")
plt.grid(alpha=0.3)

plt.savefig("validation_curve.png", dpi=150)
plt.show()
