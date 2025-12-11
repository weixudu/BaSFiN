# -*- coding: utf-8 -*-
"""
畫出 Random Search 的 (LR, KL) -> VALID AUC 3D Surface，並提供滑桿互動旋轉
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import griddata
from scipy.interpolate import Rbf

# =========================
# 解析 LOG
# =========================
def parse_log(path: str | Path) -> pd.DataFrame:
    """
    讀取並解析隨機搜尋(log)結果，回傳欄位為 ['KL', 'LR', 'valid_auc'] 的 DataFrame。
    """
    pattern_hp = re.compile(r"--\s*Sample\s+\d+/\d+:\s*KL=([\d.]+)\s+LR=([\d.]+)\s*--")
    pattern_auc = re.compile(r"Mean\s+\S*?VALID\s+AUC\s+([\d.]+)", re.IGNORECASE)

    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        current = None
        for line in fh:
            line = line.strip()

            # 超參數行
            m_hp = pattern_hp.search(line)
            if m_hp:
                kl, lr = map(float, m_hp.groups())
                current = {"KL": kl, "LR": lr}
                continue

            # AUC 行
            if current is not None:
                m_auc = pattern_auc.search(line)
                if m_auc:
                    current["valid_auc"] = float(m_auc.group(1))
                    records.append(current)
                    current = None

    if not records:
        raise ValueError("在 log 中找不到符合格式的記錄")
    return pd.DataFrame(records)


# =========================
# 繪圖
# =========================
def plot_surface_lr_kl_auc(df: pd.DataFrame,
                           elev_init: float = 30,
                           azim_init: float = 135,
                           ) :
    """
    以平滑曲面呈現 LR、KL 與 VALID AUC 之關係，並提供滑桿調整視角。
    """
    # 數據
    x, y, z = df["LR"].values, df["KL"].values, df["valid_auc"].values

    # 建立 2D 網格
    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)

    # 內插成網格上的 Z
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # rbf = Rbf(x, y, z, function='multiquadric')  # 可選: 'linear', 'cubic', 'gaussian'...
    # Zi = rbf(Xi, Yi)


    # --------- 主要圖 ---------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Xi, Yi, Zi,
                           cmap="viridis",
                           edgecolor="none",
                           alpha=0.92)
    
    # --- 加最大點標記 ---
    best_idx = df["valid_auc"].idxmax()
    best_lr  = df.loc[best_idx, "LR"]
    best_kl  = df.loc[best_idx, "KL"]
    best_auc = df.loc[best_idx, "valid_auc"]

    # ax.scatter(best_lr, best_kl, best_auc,
    #         color='red', s=60, marker='o', label='Max VALID AUC')
    # ax.text(best_lr, best_kl, best_auc + 0.002,
    #         f"Max AUC={best_auc:.4f}",
    #         color='red')

    # 標籤
    ax.set_xlabel("Learning Rate (LR)")
    ax.set_ylabel("KL Weight")
    ax.set_zlabel("Mean VALID AUC")
    ax.set_title("VALID AUC Surface: KL Weight vs. Learning Rate")

    # 初始視角
    ax.view_init(elev=elev_init, azim=azim_init)

    # Colorbar
    fig.colorbar(surf, shrink=0.55, aspect=12, pad=0.07,
                 label="Mean VALID AUC")

    # --------- 滑桿區域 ---------
    plt.subplots_adjust(bottom=0.25)  # 預留底部空間

    ax_az   = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_elev = plt.axes([0.15, 0.06, 0.7, 0.03])

    s_az   = Slider(ax_az,   "Azimuth",   0, 360, valinit=azim_init,  valstep=1)
    s_elev = Slider(ax_elev, "Elevation", -90,  90, valinit=elev_init, valstep=1)

    def update(val):
        ax.view_init(elev=s_elev.val, azim=s_az.val)
        fig.canvas.draw_idle()

    s_az.on_changed(update)
    s_elev.on_changed(update)


    plt.show()


# =========================
# 主程式
# =========================
if __name__ == "__main__":
    # === 修改這裡 ===
    log_path = "logs/BBB/BaS_random_2025-09-17_15-32-34.log"

    df = parse_log(log_path)
    print(df.head())

    plot_surface_lr_kl_auc(df,
                           elev_init=30,
                           azim_init=135)

