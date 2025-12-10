import pandas as pd

df1 = pd.read_csv('../data/boxscore/boxscore_basic_2009_2024.csv')
df2 = pd.read_csv('../data/boxscore/advance/boxscore_advance_2009_2024.csv')

# 合併表單
merged_df = pd.merge(df1, df2, on=['id', 'team', 'player_name', 'year'], how='inner')

# 顯示合併後的表單
print(merged_df.columns)
merged_df.to_csv('data_merge_2009_2024.csv', index=False)