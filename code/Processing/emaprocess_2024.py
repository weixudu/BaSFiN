import torch
import os
import numpy as np
import pandas as pd
import json
import random

# 路徑設定
tensor_path = "../data/ema_tensor/features_ema_rmse.pt"
csv_path = "../data/final_data/data_2013_2024.csv"
output_dir = "../data/ema_tensor"
mapping_path = "../data/tensor/game_id_mapping.json"

os.makedirs(output_dir, exist_ok=True)

try:
    tensor_data = torch.load(tensor_path, weights_only=False)
except Exception as e:
    print(f"載入張量失敗: {e}")
    raise

# 載入 CSV 文件
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"載入 CSV 失敗: {e}")
    raise

# 按 id（game_id）排序，確保 game_id 升序
df = df.sort_values('id').reset_index(drop=True)

# 提取原始張量中的 player_id, id, ema_features
player_ids = tensor_data[:, 0].numpy()  # [M]
ids = tensor_data[:, 1].numpy()         # [M]
ema_features = tensor_data[:, 2:]       # [M, N]
feature_dim = ema_features.shape[1]     # 特徵數，例如 32

# 將 id (game_id) 映射為連續索引，從最小 game_id 開始對應到 0
unique_ids = np.unique(ids)  # 獲取所有唯一的 game_id
unique_ids.sort()  # 按升序排序
game_id_mapping = {id_: idx for idx, id_ in enumerate(unique_ids)}  # 例如 {11582: 0, 11583: 1, ...}

# 創建反向映射，用於查找原始 id 對應的連續索引
reverse_game_id_mapping = {id_: idx for idx, id_ in enumerate(unique_ids)}

# 初始化張量
num_games = len(df)
num_players_per_game = 10
tensor_shape = (num_games, num_players_per_game, feature_dim)
game_ema_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
game_id_mapping_json = {}  # 儲存張量索引到 game_id 的對應

# 記錄缺失數據
missing_game_ids = []
missing_player_ids = []

# 填充張量
for i, row in df.iterrows():
    game_id = row['id']  # 原始 game_id，例如 11582
    game_id_index = reverse_game_id_mapping.get(game_id, -1)  # 連續索引
    game_id_mapping_json[str(i)] = str(int(game_id))  # 儲存到 JSON，例如 "0": "11582"

    if game_id_index == -1:
        missing_game_ids.append(game_id)
        game_ema_tensor[i] = torch.zeros((num_players_per_game, feature_dim), dtype=torch.float32)
        print(f"Warning: game_id {game_id} not found in tensor data")
        continue

    # 獲取該比賽的 10 個 player_id
    player_cols = [f'player{j+1}' for j in range(num_players_per_game)]
    game_player_ids = [row[col] if pd.notna(row[col]) and row[col] != -1 else -1 for col in player_cols]

    # 從原始張量中提取該 game_id 的數據
    game_mask = ids == game_id  # 找到對應 game_id 的行
    game_data = tensor_data[game_mask]  # [num_players, 2 + feature_dim]

    # 按 CSV 的 player_id 順序填充
    for j, player_id in enumerate(game_player_ids):
        player_id = float(player_id) if player_id != -1 else -1.0
        if player_id == -1.0:
            missing_player_ids.append((game_id, player_id))
            game_ema_tensor[i, j] = torch.zeros(feature_dim, dtype=torch.float32)
            continue

        # 查找該 player_id 在 game_data 中的數據
        player_mask = game_data[:, 0] == player_id
        if player_mask.sum() == 0:
            missing_player_ids.append((game_id, player_id))
            game_ema_tensor[i, j] = torch.zeros(feature_dim, dtype=torch.float32)
            print(f"Warning: player_id {player_id} not found in game_id {game_id}")
            continue

        # 提取特徵（忽略 player_id 和 game_id）
        player_features = game_data[player_mask, 2:][0]  # 取第一筆（假設唯一）
        game_ema_tensor[i, j] = player_features

# 加權處理
mp_index = 20  # MP_min_ema 的索引，根據 ordered_features 調整
if mp_index >= feature_dim:
    raise ValueError(f"mp_index {mp_index} 超出特徵範圍 (0 to {feature_dim-1})")

MP_min_ema = game_ema_tensor[:, :, mp_index].unsqueeze(-1)  # [num_games, 10, 1]
weights = (MP_min_ema / 48.0).clip(min=0.0)  # [num_games, 10, 1]
weights = 1
game_ema_tensor = game_ema_tensor * weights  # 應用加權

# 標準化
do_standardization = True
if do_standardization:
    feature_means = game_ema_tensor.view(-1, feature_dim).mean(dim=0)  # [feature_dim]
    feature_stds = game_ema_tensor.view(-1, feature_dim).std(dim=0)    # [feature_dim]
    feature_stds[feature_stds == 0] = 1.0  # 避免除以 0
    game_ema_tensor = (game_ema_tensor - feature_means) / feature_stds

# 保存張量和 game_id 對應表
out_path = os.path.join(output_dir, "ematensor_test.pt")
torch.save(game_ema_tensor, out_path)
with open(mapping_path, 'w') as f:
    json.dump(game_id_mapping_json, f)

# 打印結果
print(f"處理後的張量已儲存至 {out_path}")
print(f"Game ID 對應表已儲存至 {mapping_path}")
print(f"張量形狀: {game_ema_tensor.shape}")
print(f"張量結構: [game_id, [player_id, [ema_features]]]")
print(f"特徵數: {feature_dim}")
print(f"Missing game IDs: {len(missing_game_ids)}")
if missing_game_ids:
    print(f"Sample missing game IDs (up to 5): {missing_game_ids[:5]}")
print(f"Missing player IDs: {len(missing_player_ids)}")
if missing_player_ids:
    print(f"Sample missing player IDs (up to 5): {missing_player_ids[:5]}")

# 打印 player_id 和 game_id 範圍
unique_player_ids = np.unique([pid for pid in player_ids if pid != -1])
print(f"\nPlayer ID 範圍: [{np.min(unique_player_ids):.0f}, {np.max(unique_player_ids):.0f}]")
print(f"Game ID 範圍 (連續索引): [0, {num_games-1}]")
print(f"原始 Game ID 範圍: [{np.min(unique_ids):.0f}, {np.max(unique_ids):.0f}]")

# 檢查 game_id_mapping_json 的鍵
print(f"game_id_mapping_json 鍵數量: {len(game_id_mapping_json)}")
print(f"game_id_mapping_json 鍵範圍: [{min(game_id_mapping_json.keys())}, {max(game_id_mapping_json.keys())}]")

# 隨機打印一筆資料的 ema_features
random_game_idx = random.randint(0, num_games-1)
random_player_idx = random.randint(0, num_players_per_game-1)
while game_ema_tensor[random_game_idx, random_player_idx].sum() == 0:  # 確保不選到全 0 向量
    random_game_idx = random.randint(0, num_games-1)
    random_player_idx = random.randint(0, num_players_per_game-1)

random_features = game_ema_tensor[random_game_idx, random_player_idx].numpy()
random_game_id = game_id_mapping_json.get(str(random_game_idx), "Unknown")
random_player_id = df.iloc[random_game_idx][f'player{random_player_idx+1}']
print(f"\n隨機選擇的資料:")
print(f"Game ID (連續索引): {random_game_idx}, 原始 Game ID: {random_game_id}")
print(f"Player ID: {random_player_id:.0f}")
print(f"EMA Features (標準化後):")
for i, feat in enumerate(random_features):
    print(f"  Feature {i}: {feat:.4f}")