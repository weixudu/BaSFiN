import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(df, label):
    output_dir = "../plot"
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    x = df['Games']
    y = df['Win_Rate']
    colors = df['Skill_Mu']  
    original_cmap = plt.get_cmap('RdYlBu')
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_RdYlBu", original_cmap(np.linspace(0.2, 0.95, 256))
    )

    vmin = df['Skill_Mu'].min() - 0.5  
    vmax = df['Skill_Mu'].max() + 0.5  
    scatter = ax.scatter(x, y, c=colors, cmap='RdYlBu', vmin=vmin, vmax=vmax,s=40, alpha=1, label=label)

    ax.set_xlabel('Games')
    ax.set_ylabel('Win Rate')
    ax.set_title(f'{label} - 2D Plot: Games vs Win Rate (Color = Skill Mu)')
    ax.legend()
    ax.grid(True)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Skill Mu')
    plt.show()
    # plt.savefig(f"{output_dir}/{label}.png")
    # plt.close()

def main():
    files = {
        "player_stats_BaS": "../output/BaS/player_stats_2025-07-13_21-00-10.csv",
        "player_stats_BT": "../output/player_stats_BT.csv"
    }

    for label, path in files.items():
        try:
            df = pd.read_csv(path)
            plot_stats(df, label)
            print(f"已完成 {label} 的圖形繪製與儲存。")
        except Exception as e:
            print(f"處理 {label} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
