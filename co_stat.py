import os
import re
import pandas as pd

log_file_path = 'logs/final_logs/Search/cofim_0.0001p2_random_search.log'

if not os.path.exists(log_file_path):
    raise FileNotFoundError(f"Log file not found at: {log_file_path}")

with open(log_file_path, 'r', encoding='utf-8') as f:
    logs_text = f.read()

# 正則表達式，匹配超參數和 AUC
pattern = re.compile(
    r"Player Dim:\s*(\d+),\s*"
    r"Hidden Dim:\s*(\d+),\s*"
    r"Need Att:\s*(True|False),\s*"
    r"Dropout:\s*(\d*\.\d+),\s*"
    r"MLP Hidden:\s*(\d+),\s*"
    r"Weight Decay:\s*(\d*\.\d+),\s*"
    r"Average Valid AUC(?::\s*(\d*\.\d+))?"
)

data = []
for line in logs_text.split('\n'):
    match = pattern.search(line)
    if match:
        (
            pdim,              # Player Dim
            hdim,              # Hidden Dim
            need_att_str,      # Need Att (True / False)
            dropout,           # Dropout
            mlp,               # MLP Hidden
            weight_decay,      # Weight Decay
            auc                # Average Valid AUC (可能是 None)
        ) = match.groups()

        data.append({
            'player_dim':      int(pdim),
            'hidden_dim':      int(hdim),
            'need_att':        need_att_str == 'True',
            'dropout':         float(dropout),
            'mlp_hidden':      int(mlp),
            'weight_decay':    float(weight_decay),
            'auc':             float(auc) if auc is not None else None
        })

df = pd.DataFrame(data)
df_sorted = df.sort_values(by='auc', ascending=False).reset_index(drop=True)

top_n = 5
df_top_n = df_sorted.head(top_n)

# # 輸出結果
# print("▶ 所有組合最大與最小值")
# print(df_sorted.to_string(index=False))

print(f"\n▶ Top {top_n} 組合")
print(df_top_n.to_string(index=False))

if not df_sorted.empty and 'auc' in df_sorted.columns:
    valid_aucs = df_sorted['auc'].dropna()
    if not valid_aucs.empty:
        max_auc = valid_aucs.max()
        min_auc = valid_aucs.min()
        print(f"\n▶ 最大 AUC: {max_auc:.4f}")
        print(f"▶ 最小 AUC: {min_auc:.4f}")
    else:
        print("\n▶ 無有效 AUC 數據（全為 NaN）")
else:
    print("\n▶ 無有效 AUC 數據")