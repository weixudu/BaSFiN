import os
import re
import pandas as pd

log_file_path = 'logs/final_logs/Search/FiN_pairwise_0.0001p2_random_search.log'

if not os.path.exists(log_file_path):
    raise FileNotFoundError(f"Log file not found at: {log_file_path}")

logs_text = []
try:
    with open(log_file_path, 'r', encoding='utf-8-sig') as f:  
        logs_text = [line.strip() for line in f if line.strip()]  
except UnicodeDecodeError as e:
    print(f"UTF-8 解碼失敗，嘗試使用 latin1 編碼: {e}")
    with open(log_file_path, 'r', encoding='latin1') as f:
        logs_text = [line.strip() for line in f if line.strip()]

pattern = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - "
    r"Player Dim:\s*(\d+),\s*"
    r"Intermediate Dim:\s*(\d+),\s*"
    r"Dropout:\s*(\d*\.\d*),\s*"  
    r"Need Att:\s*(True|False),\s*"
    r"MLP Hidden:\s*(\d+),\s*"
    r"Average Valid AUC:\s*(\d*\.\d*)"  
)

data = []
found_summary = False
for i, line in enumerate(logs_text):
    if "Summary of best results per combination:" in line:  
        found_summary = True
        print(f"找到 Summary 標誌行 (行 {i+1}): {line}")
        continue
    if found_summary:
        match = pattern.search(line)
        if match:
            try:
                (
                    pdim,           # Player Dim
                    idim,           # Intermediate Dim
                    dropout,        # Dropout
                    need_att_str,   # Need Att (True / False)
                    mlp,            # MLP Hidden
                    auc             # Average Valid AUC
                ) = match.groups()

                # 轉換數據類型
                data.append({
                    'player_dim':       int(pdim),
                    'intermediate_dim': int(idim),
                    'dropout':          float(dropout) if dropout != '.' else 0.0,
                    'need_att':         need_att_str == 'True',
                    'mlp_hidden':       int(mlp),
                    'auc':              float(auc) if auc != '.' else 0.0
                })
            except ValueError as e:
                print(f"數據轉換錯誤在行 {i+1}: {line}, 錯誤: {e}")
        else:
            print(f"未匹配的行 {i+1}: {line}")

if not data:
    print("警告：未提取到任何有效數據，請檢查日誌格式")
    print("日誌前 5 行：")
    for i, line in enumerate(logs_text[:5]):
        print(f"行 {i+1}: {line}")
    raise ValueError("日誌文件中未找到任何有效數據")

df = pd.DataFrame(data)
df_sorted = df.sort_values(by='auc', ascending=False).reset_index(drop=True)

top_n = 5
df_top_n = df_sorted.head(top_n)

df_sorted['auc'] = df_sorted['auc'].round(4)
df_top_n['auc'] = df_top_n['auc'].round(4)

# # 輸出結果
# print("\n▶ 所有組合與 AUC（排序後）")
# print(df_sorted.to_string(index=False))

print(f"\n▶ Top {top_n} 組合")
print(df_top_n.to_string(index=False))

# 打印最大和最小 AUC
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
