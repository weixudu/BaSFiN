import re
import matplotlib.pyplot as plt

# 兩個 MLP 訓練策略的 log 檔案路徑
log_path_frozen = "logs/NAC+/BaSFiN_noInter_20250708_004154.log"
log_path_e2e = "logs/NAC+/BaSFiN_noInter_noFreeze_20250708_004154.log"

# 兩條線性 baseline 的 AUC
linear_auc_frozen = 0.6829
linear_auc_e2e = 0.6723

def parse_log_file(filepath):
    """
    解析 log 檔案，取得 Prob Dim 與 Average Valid AUC
    """
    prob_dims = []
    valid_aucs = []
    with open(filepath, 'r') as file:
        for line in file:
            dim_match = re.search(r'Prob Dim:\s*(\d+)', line)
            auc_match = re.search(r'Average Valid AUC:\s*([0-9.]+)', line)
            if dim_match and auc_match:
                prob_dims.append(int(dim_match.group(1)))
                valid_aucs.append(float(auc_match.group(1)))
    return prob_dims, valid_aucs

# 讀取兩個 MLP 訓練策略的資料
dims_frozen, aucs_frozen = parse_log_file(log_path_frozen)
dims_e2e, aucs_e2e = parse_log_file(log_path_e2e)

# 畫圖
plt.figure(figsize=(10, 6))

# MLP 非線性版本
plt.plot(
    dims_frozen,
    aucs_frozen,
    linestyle='-',
    linewidth=2,
    label='Frozen + MLP'
)

plt.plot(
    dims_e2e,
    aucs_e2e,
    linestyle='-',
    linewidth=2,
    label='End-to-End + MLP'
)

# Linear baseline 畫成相同維度範圍的平線
plt.plot(
    dims_frozen,
    [linear_auc_frozen] * len(dims_frozen),
    color='red',
    linestyle='--',
    linewidth=2,
    label='Frozen + Linear'
)

plt.plot(
    dims_e2e,
    [linear_auc_e2e] * len(dims_e2e),
    color='orange',
    linestyle='--',
    linewidth=2,
    label='End-to-End + Linear'
)


# 軸設定
plt.xlabel('Probabilistic Latent Dimension', fontsize=12)
plt.ylabel('Validation AUC', fontsize=12)

# 標題
plt.title('Validation AUC Comparison Across Model Types and Training Strategies', fontsize=14)

# y 軸範圍
plt.ylim(0.6, 0.7)

# 美觀設定
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()