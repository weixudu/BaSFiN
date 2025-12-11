import os
import pandas as pd


folder_path = r'C:\Users\Rain\Desktop\碩論\code'
output_file = 'combined_boxscore_2009_2014_with_year.csv'


csv_files = sorted([f for f in os.listdir(folder_path) if f.startswith('boxscore_') and f.endswith('.csv')])
combined_df = pd.DataFrame()

# 讀取每個CSV並合併到一起
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    # 從檔案名稱中提取年份（檔案名格式為 boxscore_XXXX.csv）
    year = file.split('_')[2].split('.')[0]
    
    print(f"正在讀取並合併 {file} (年份: {year})...")
    df = pd.read_csv(file_path)
    
    # 在 DataFrame 中新增一個名為 'year' 的欄位，填入對應的年份
    df['year'] = int(year)
    combined_df = pd.concat([combined_df, df], ignore_index=True)


output_path = os.path.join(folder_path, output_file)
combined_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"所有資料已合併並儲存至 {output_file}")
