# -*- coding: utf-8 -*-
"""
sigma_dual_xaxis_plot_split.py   (English labels, two groups)

差異
----
1. 將追蹤名單拆成兩組 (group1, group2)
2. 迴圈一次產生兩張圖
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ─── configurable section ────────────────────────────────────────────────
CSV_PATH   = "../output/BaS/sigma_tracking_stage2_best_trial_2025-06-27_14-41-33.csv"

GROUP1_IDS = [132, 438, 1, 355, 544]              # 第一張圖
GROUP2_IDS = [1147, 1168, 1019, 876, 148]         # 第二張圖

YSCALE_LOG = True
SAVE_DIR   = Path("../output/BaS")                 # 存檔目錄
# ─────────────────────────────────────────────────────────────────────────

# 讀檔並計算 step
df = (pd.read_csv(CSV_PATH)
        .query("player_id in @GROUP1_IDS or player_id in @GROUP2_IDS")
        .sort_values(["epoch", "batch"])
        .reset_index(drop=True))

B_max = df["batch"].max()
df["step"] = (df["epoch"] - 1) * B_max + df["batch"]

# 方便重複使用的繪圖函式 ---------------------------------------------------
def plot_sigma(sub_ids, title_suffix, file_suffix):
    fig, ax = plt.subplots(figsize=(11, 5))

    for pid in sub_ids:
        sub = df[df["player_id"] == pid]
        ax.plot(sub["step"], sub["sigma"], label=f"Player {pid}")

    # bottom axis
    ax.set_xlabel("Training step (Epoch × Batch index)")
    ax.set_ylabel("Sigma")

    if YSCALE_LOG:
        ax.set_yscale("log")

        def y_fmt(y, _):
            return f"{y:.1f}" if y >= 1 else f"{y:.2f}"
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
    else:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.set_title(f"Sigma Convergence Trend – {title_suffix}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    ax.legend()

    # top axis (epoch)
    def step2epoch(step):  return (step - 1) // B_max + 1
    def epoch2step(epoch): return (epoch - 1) * B_max + 1

    secax = ax.secondary_xaxis("top", functions=(step2epoch, epoch2step))
    secax.set_xlabel("Epoch")

    max_epoch = df["epoch"].max()
    epoch_ticks = [epoch2step(e) for e in range(1, max_epoch + 1)]

    ax.set_xticks(epoch_ticks)
    ax.set_xticklabels([str(e) for e in range(1, max_epoch + 1)])
    secax.set_xticks(epoch_ticks)
    secax.set_xticklabels([""] + [str(e) for e in range(2, max_epoch + 1)])

    plt.tight_layout()

    if SAVE_DIR:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAVE_DIR / f"sigma_convergence_{file_suffix}.png"
        plt.savefig(out_path, dpi=300)
        print(f"Figure saved to {out_path}")

    plt.show()


# ─── produce the two figures ────────────────────────────────────────────
plot_sigma(GROUP1_IDS, "Group 1", "group1")
plot_sigma(GROUP2_IDS, "Group 2", "group2")
