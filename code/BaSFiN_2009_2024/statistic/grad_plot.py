#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_grad_trends.py
讀取 grads_all_freeze.csv，為 Stage 0 與 Stage 1 繪製：
  • 左軸：三大模組梯度 (log10 scale)
  • 右軸：AUC 走勢 (黑色虛線)
  • 只看 split=Valid
  • 每張圖自動另存成 PNG 檔
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
CSV_PATH = "../data/grads_all.csv"      # 你的 freeze 版 CSV
MODULES  = ["modgrad_z", "modgrad_comp", "modgrad_coop"]
METRIC   = "auc"                       # 可改成 acc、logloss
SPLIT    = "Valid"                     # 只選這種 split
COLORS   = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# ---------------------------------------------------------

def main() -> None:
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"❌ 找不到檔案：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["split"] == SPLIT]

    if df.empty:
        raise ValueError(f"❌ 找不到 split = {SPLIT} 的紀錄，請檢查 CSV。")

    # 先 groupby 同一 stage/epoch 平均
    agg_dict = {m: "mean" for m in MODULES} | {METRIC: "mean"}
    df_avg = (
        df.groupby(["stage", "epoch"])
          .agg(agg_dict)
          .reset_index()
          .sort_values(["stage", "epoch"])
    )

    for stage in sorted(df_avg["stage"].unique()):
        sub = df_avg[df_avg["stage"] == stage]

        fig, ax_grad = plt.subplots(figsize=(8, 5))

        # --- 左軸：三條模組梯度（log10） ---
        for idx, m in enumerate(MODULES):
            ax_grad.plot(
                sub["epoch"], sub[m],
                marker="o", label=m,
                linewidth=2, color=COLORS[idx]
            )
        ax_grad.set_xlabel("Epoch")
        ax_grad.set_ylabel("Gradient Magnitude (log10 scale)")
        ax_grad.set_yscale("log")
        ax_grad.grid(True, alpha=0.3)

        # --- 右軸：AUC ---
        ax_auc = ax_grad.twinx()
        ax_auc.plot(
            sub["epoch"], sub[METRIC],
            marker="s", linestyle="--", linewidth=2,
            color="black", label=METRIC.upper()
        )
        ax_auc.set_ylabel(METRIC.upper())

        ymin, ymax = sub[METRIC].min(), sub[METRIC].max()
        ax_auc.set_ylim(ymin - 0.02, ymax + 0.02)

        # --- Legend 合併 ---
        lines1, labels1 = ax_grad.get_legend_handles_labels()
        lines2, labels2 = ax_auc.get_legend_handles_labels()
        ax_grad.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper center", bbox_to_anchor=(0.5, 1.15),
            ncol=4, frameon=False
        )

        plt.title(f"Stage {stage}: Module Gradients (log) & {METRIC.upper()} Trend")

        plt.tight_layout()
        outname = f"stage{stage}_grad_auc_noFreeze.png"
        plt.savefig(outname, dpi=300)
        print(f"✅ 已儲存圖檔：{outname}")

        plt.show()

if __name__ == "__main__":
    main()
