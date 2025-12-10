import os
import re
import pandas as pd

log_file_path = 'logs/final_logs/Search/BaSFiN_random_search_20250622_112530.log'
if not os.path.exists(log_file_path):
    raise FileNotFoundError(log_file_path)

with open(log_file_path, 'r', encoding='utf-8') as f:
    logs_text = f.read()

# ——— 新格式 regex ——————————————————————————————————————
PATTERN = re.compile(
    r"Prob Dim:\s*(?P<prob>\d+),\s*"
    r"Dropout:\s*(?P<dropout>[\d.]+),\s*"
    r"Learning Rate:\s*(?P<lr>[\deE\-\.+]+),\s*"
    r"Average Valid AUC:\s*(?P<auc>[\d.]+)"
)

records = []
for line in logs_text.splitlines():
    m = PATTERN.search(line)
    if not m:
        continue
    gd = m.groupdict()
    records.append({
        "prob_dim": int(gd["prob"]),
        "dropout":  float(gd["dropout"]),
        "lr":       float(gd["lr"]),
        "auc":      float(gd["auc"]),
    })

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("Log 內未匹配任何記錄")

df_sorted = df.sort_values("auc", ascending=False).reset_index(drop=True)
top_n = 5
df_top = df_sorted.head(top_n)

# ——— 輸出 ——————————————————————————————————————————————
print(f"\n▶ Top {top_n} 組合")
print(df_top.to_string(index=False))

valid_aucs = df_sorted["auc"].dropna()
print(f"\n▶ 最大 AUC: {valid_aucs.max():.4f}")
print(f"▶ 最小 AUC: {valid_aucs.min():.4f}")
