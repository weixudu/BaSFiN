import pandas as pd
import os

# 讀取 CSV 檔案
df = pd.read_csv('../data/data_merge_2009_2024.csv')

# 設定起始與結束年份
start_year = 2009
end_year = 2024
window_size = 12

# 建立存放資料夾的路徑
output_dir = '../data/feature_csv'

# 確保資料夾存在，若無則建立
os.makedirs(output_dir, exist_ok=True)

# 建立分割的子集資料
for start in range(start_year, end_year - window_size + 2):
    end = start + window_size - 1
    subset = df[(df['year'] >= start) & (df['year'] <= end)]
    
    # 設定包含路徑的檔案名稱
    filename = f'data_merge_{start}_{end}.csv'
    filepath = os.path.join(output_dir, filename)
    
    # 儲存子資料
    subset.to_csv(filepath, index=False)
    
    # 印出子集資訊
    print(f"Created: {filepath}, rows: {len(subset)}")
