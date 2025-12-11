# import pandas as pd
# import plotly.graph_objects as go

# nba_games_result = pd.read_csv('data/nba_games_result.csv')
# box_team = pd.read_csv('data/boxscore/box_team.csv')

# # 創建全名和縮寫的對應字典
# team_name_mapping = {
#     "Atlanta Hawks": "ATL",
#     "Boston Celtics": "BOS",
#     "Brooklyn Nets": "BRK",
#     "Charlotte Hornets": "CHO",
#     "Chicago Bulls": "CHI",
#     "Cleveland Cavaliers": "CLE",
#     "Dallas Mavericks": "DAL",
#     "Denver Nuggets": "DEN",
#     "Detroit Pistons": "DET",
#     "Golden State Warriors": "GSW",
#     "Houston Rockets": "HOU",
#     "Indiana Pacers": "IND",
#     "Los Angeles Clippers": "LAC",
#     "Los Angeles Lakers": "LAL",
#     "Memphis Grizzlies": "MEM",
#     "Miami Heat": "MIA",
#     "Milwaukee Bucks": "MIL",
#     "Minnesota Timberwolves": "MIN",
#     "New Orleans Pelicans": "NOP",
#     "New York Knicks": "NYK",
#     "Oklahoma City Thunder": "OKC",
#     "Orlando Magic": "ORL",
#     "Philadelphia 76ers": "PHI",
#     "Phoenix Suns": "PHO",
#     "Portland Trail Blazers": "POR",
#     "Sacramento Kings": "SAC",
#     "San Antonio Spurs": "SAS",
#     "Toronto Raptors": "TOR",
#     "Utah Jazz": "UTA",
#     "Washington Wizards": "WAS"
# }

# team_1_fullname = 'Boston Celtics'
# team_2_fullname = 'Cleveland Cavaliers'

# team_1 = team_name_mapping[team_1_fullname]
# team_2 = team_name_mapping[team_2_fullname]

# team_games = nba_games_result[((nba_games_result['Visitor'] == team_1_fullname) & (nba_games_result['Home'] == team_2_fullname)) |
#                               ((nba_games_result['Visitor'] == team_2_fullname) & (nba_games_result['Home'] == team_1_fullname))]

# def get_team_ratings(team_name):
#     team_data = box_team[box_team['team'] == team_name]
#     return team_data['ORtg'].values[0], team_data['DRtg'].values[0], team_data['Nrtg'].values[0]
# team_1_ortg, team_1_drtg, team_1_nrtg = [], [], []
# team_2_ortg, team_2_drtg, team_2_nrtg = [], [], []
# game_ids = team_games['id'].tolist()
# for _, row in team_games.iterrows():
#     if row['Visitor'] == team_1_fullname:
#         ortg_1, drtg_1, nrtg_1 = get_team_ratings(team_1)
#         ortg_2, drtg_2, nrtg_2 = get_team_ratings(team_2)
#     else:
#         ortg_1, drtg_1, nrtg_1 = get_team_ratings(team_2)
#         ortg_2, drtg_2, nrtg_2 = get_team_ratings(team_1)
    
#     team_1_ortg.append(ortg_1)
#     team_1_drtg.append(drtg_1)
#     team_1_nrtg.append(nrtg_1)
    
#     team_2_ortg.append(ortg_2)
#     team_2_drtg.append(drtg_2)
#     team_2_nrtg.append(nrtg_2)

# team_games['Winner'] = team_games.apply(lambda row: row['Visitor'] if row['Visitor PTS'] > row['Home PTS'] else row['Home'], axis=1)
# fig = go.Figure()

# # ORtg 折線圖
# fig.add_trace(go.Scatter(x=game_ids, y=team_1_ortg, 
#                          mode='lines+markers', name=f'{team_1_fullname} ORtg',
#                          text=[f'Game {id}: Winner: {row["Winner"]}' for id, row in zip(game_ids, team_games.iterrows())],
#                          hoverinfo='text'))
# fig.add_trace(go.Scatter(x=game_ids, y=team_2_ortg, 
#                          mode='lines+markers', name=f'{team_2_fullname} ORtg',
#                          text=[f'Game {id}: Winner: {row["Winner"]}' for id, row in zip(game_ids, team_games.iterrows())],
#                          hoverinfo='text'))

# fig.update_layout(
#     title="ORtg Comparison",
#     xaxis_title="Game ID",
#     yaxis_title="ORtg",
#     hovermode="closest"
# )

# # 顯示圖表
# fig.show()


# import os
# import pandas as pd
# import numpy as np
# import random
# import json

# high_mse_features = ['ORtg', 'DRtg', 'NRtg', 'AST%', 'USG%', 'TOV%', 'BPM', '+/-']

# # -------------------------------------------------------------------
# # 2. 讀取資料並進行 player_id 映射
# # -------------------------------------------------------------------
# data_path = "../data/year_merge/data_merge_2018_2024.csv"
# mapping_path = "../data/player_id_mapping_2009_2024.csv"

# # 讀取主資料
# df = pd.read_csv(data_path)

# # 讀取 player_id 映射
# player_mapping = pd.read_csv(mapping_path)
# player_id_dict = dict(zip(player_mapping['player_name'], player_mapping['player_id']))

# # 將 player_name 替換為 player_id
# df['player_id'] = df['player_name'].map(player_id_dict)
# if df['player_id'].isna().any():
#     print("警告：部分 player_name 無法映射到 player_id，可能需檢查映射檔案")
#     df = df.dropna(subset=['player_id'])

# # 依 player_id 和 id 排序
# df = df.sort_values(['player_id', 'id']).reset_index(drop=True)

# # 填補數值欄位遺失值為 0
# df[high_mse_features] = df[high_mse_features].fillna(0)

# players = df['player_id'].unique()
# random.seed(42)  # 確保可重現
# selected_players = random.sample(list(players), 3)
# print(f"選取的球員 ID：{selected_players}")

# # -------------------------------------------------------------------
# # 4. 為每個高 MSE 特徵生成折線圖資料 (JSON 格式)
# # -------------------------------------------------------------------
# chart_configs = {}

# for feature in high_mse_features:
#     datasets = []
#     max_length = 0

#     for player_id in selected_players:
#         player_df = df[df['player_id'] == player_id].copy()
#         player_df = player_df.reset_index(drop=True)
#         player_df['continuous_id'] = range(1, len(player_df) + 1)

#         max_length = max(max_length, len(player_df))

#         datasets.append({
#             'label': f'Player {player_id}',
#             'data': player_df[feature].tolist(),
#             'borderColor': f'rgb({random.randint(100, 255)}, {random.randint(100, 255)}, {random.randint(100, 255)})',
#             'backgroundColor': 'rgba(0, 0, 0, 0)',
#             'fill': False
#         })

#     chart_config = {
#         "type": "line",
#         "data": {
#             "labels": [str(i) for i in range(1, max_length + 1)],
#             "datasets": datasets
#         },
#         "options": {
#             "responsive": True,
#             "plugins": {
#                 "title": {
#                     "display": True,
#                     "text": f"{feature} for Selected Players"
#                 },
#                 "legend": {
#                     "position": "top"
#                 }
#             },
#             "scales": {
#                 "x": {
#                     "title": {
#                         "display": True,
#                         "text": "Game ID (Continuous)"
#                     }
#                 },
#                 "y": {
#                     "title": {
#                         "display": True,
#                         "text": feature
#                     }
#                 }
#             }
#         }
#     }

#     # 儲存到字典中，方便匯出或展示
#     chart_configs[feature] = chart_config

# # Optional: 輸出為 JSON 檔案
# with open("chart_configs.json", "w", encoding='utf-8') as f:
#     json.dump(chart_configs, f, ensure_ascii=False, indent=2)

# print("圖表配置已成功產生並儲存至 chart_configs.json")



import json
import matplotlib.pyplot as plt

# 讀取 Chart.js 格式的 JSON 檔案
with open('chart_configs.json', 'r', encoding='utf-8') as f:
    chart_configs = json.load(f)

# 為每個 feature 畫圖
for feature, config in chart_configs.items():
    labels = config['data']['labels']
    datasets = config['data']['datasets']

    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        plt.plot(
            labels[:len(dataset['data'])],  # 有些資料長度不一致
            dataset['data'],
            label=dataset['label']
        )

    # 設定圖表標題與標籤
    plt.title(config['options']['plugins']['title']['text'])
    plt.xlabel(config['options']['scales']['x']['title']['text'])
    plt.ylabel(config['options']['scales']['y']['title']['text'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
