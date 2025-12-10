#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_paramgrad_broken_axis.py
  • 左側是 Param Grad 的「斷開 Y 軸河流圖」
  • 下軸放大顯示 0–1 區間
  • 上軸顯示 5–70
  • 圖例在下軸右下角
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 設定 ----------
CSV_PATH   = "../output/grads_all_no_freeze_trials.csv"
GRADS      = ["paramgrad_bbb", "paramgrad_fi", "paramgrad_anfm"]
METRIC     = "TESTAUC"
USE_LOG10  = False
DISPLAY_LABELS = [
    "BaS Module",
    "Competition Module",
    "Cooperation Module"
]

COLORS     = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def main() -> None:
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")

    # 讀取 CSV
    df = pd.read_csv(CSV_PATH)

    # 每 stage×epoch 取 trial 平均
    mean_df = (
        df.groupby(["stage", "epoch"])[GRADS + [METRIC]]
          .mean()
          .reset_index()
          .sort_values(["stage", "epoch"])
    )

    # Stage 1 epoch 接到 Stage 0 後面
    stage0 = mean_df[mean_df.stage == 0].copy()
    stage1 = mean_df[mean_df.stage == 1].copy()
    offset = stage0["epoch"].max()
    stage1["epoch"] += offset
    plot_df = pd.concat([stage0, stage1], ignore_index=True)

    # X 軸
    epochs = plot_df["epoch"]

    # Y
    Y = plot_df[GRADS].copy()
    if USE_LOG10:
        Y = np.log10(Y + 1e-8)

    # ---------- 平均占比打印 ----------
    mean_values = Y.mean()
    total_mean = mean_values.sum()
    percent = mean_values / total_mean * 100

    print("\n=== 平均 ParamGrad 占比 (%) ===")
    for g in GRADS:
        print(f"{g:15s}: {percent[g]:6.2f} %")

    # ---------- Broken Axis 圖 ----------
    fig, (ax_high, ax_low) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(10, 6)
    )

    # Stackplot 兩段都畫
    for ax in (ax_low, ax_high):
        ax.stackplot(
            epochs,
            Y.T,
            colors=COLORS[:len(GRADS)],
            labels=DISPLAY_LABELS,
            alpha=0.7
        )

    # 下軸 → 放大 0–1
    ax_low.set_ylim(0, 1)

    # 上軸 → 固定 5–70
    ax_high.set_ylim(10, 80)

    # Y 軸標籤
    ax_high.set_ylabel("Parameter Gradient Contribution (%)")


    # Stage 分界線 + 文字
    # 只在上軸畫分段線和文字
    ax_high.axvline(x=offset + 0.5, color="gray", linestyle="--", linewidth=1)
    ax_high.text(offset/2,
                  ax_high.get_ylim()[1]*0.95, "stage1",
                  ha="center", va="top", fontsize=10)
    ax_high.text(offset + (plot_df["epoch"].max()-offset)/2,
                  ax_high.get_ylim()[1]*0.95, "final retraining",
                  ha="center", va="top", fontsize=10)

    for ax in (ax_low, ax_high):
        ax.grid(alpha=0.3)

    # 移除多餘邊界
    ax_low.spines['top'].set_visible(False)
    ax_high.spines['bottom'].set_visible(False)
    ax_low.tick_params(labeltop=False)
    ax_high.tick_params(labelbottom=False)

    # 斜線標記斷軸
    d = .015
    kwargs = dict(transform=ax_low.transAxes, color='k', clip_on=False)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_high.transAxes)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)


    # Legend → 只放下軸，右下角
    ax_low.legend(
        loc="lower right",
        frameon=True,
        title="Module",
        fontsize=9,
        title_fontsize=10
    )


    # Title & Layout
    fig.suptitle("Parameter Gradient Contribution over Epochs (Stage-1 → Final Retraining)")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
