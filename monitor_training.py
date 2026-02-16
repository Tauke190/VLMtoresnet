# make a validation accuracy plot
# refresh plot every validation epoch
# detect plateau
# warn if >24 hrs
# color code convergence

# RUN THIS ALONGSIDE THE TRAINING IN ANOTHER TERMINAL

import pandas as pd
import matplotlib.pyplot as plt
import time
import os

CSV_PATH = "checkpoints/fastvit-proj/summary.csv"
PLOT_PATH = "validation_live.png"

MAX_HOURS = 24
PLATEAU_EPOCHS = 5   # no improvement for N epochs -> warn


start_time = time.time()
best_acc = -1
last_improve_epoch = 0


def load_clean_csv():
    df = pd.read_csv(CSV_PATH)

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["eval_top1"] = pd.to_numeric(df["eval_top1"], errors="coerce")

    df = df.dropna(subset=["epoch", "eval_top1"])
    return df


while True:

    if not os.path.exists(CSV_PATH):
        time.sleep(60)
        continue

    df = load_clean_csv()

    if len(df) == 0:
        time.sleep(60)
        continue

    acc = df["eval_top1"]
    epochs = df["epoch"]

    current_best = acc.max()
    best_epoch = epochs.iloc[acc.idxmax()]

    # detect improvement
    if current_best > best_acc:
        best_acc = current_best
        last_improve_epoch = best_epoch

    plateau = (best_epoch - last_improve_epoch) >= PLATEAU_EPOCHS

    hours = (time.time() - start_time) / 3600
    overtime = hours > MAX_HOURS

    # -------- Plot --------
    plt.figure(figsize=(10,6))

    color = "green"

    if plateau:
        color = "orange"

    if overtime:
        color = "red"

    plt.plot(epochs, acc, marker="o", color=color)

    plt.axhline(best_acc, linestyle="--")

    title = f"Best={best_acc:.2f}% @ epoch {int(best_epoch)}"

    if plateau:
        title += " ⚠️ Plateau detected"

    if overtime:
        title += " 🔴 >24hrs"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.grid(alpha=0.3)

    plt.savefig(PLOT_PATH, dpi=140)
    plt.close()

    print(f"Updated plot - best={best_acc:.2f}%")
    
    # OPTIONAL HARD STOP WARNING
    if overtime and plateau:
        print("\n🔥 TRAINING LIKELY WASTING GPU TIME 🔥")
        print("Consider killing this job.\n")

    time.sleep(300)  # refresh every 5 min
