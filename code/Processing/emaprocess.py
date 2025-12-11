import torch
import os
import numpy as np
import pandas as pd
import json
import random

# 建立輸出資料夾
output_dir = "../data/ema_tensor"
os.makedirs(output_dir, exist_ok=True)

# 需要處理的年份尾碼
target_years = ["2020", "2021", "2022", "2023", "2024"]

for year_suffix in target_years:
    print("\n===================================================")
    print(f"✅ 開始處理年份：{year_suffix}")
    print("===================================================")

    # 自動產生路徑
    tensor_path = f"../data/ema_tensor/features_ema_rmse{year_suffix}.pt"
    csv_path = f"../data/final_data/data_{int(year_suffix)-11}_{year_suffix}.csv"
    mapping_path = f"../data/tensor/game_id_mapping_{year_suffix}.json"

    # 嘗試載入張量
    try:
        tensor_data = torch.load(tensor_path, weights_only=False)
    except Exception as e:
        print(f"❌ 載入張量失敗: {e}")
        continue

    # 載入 CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 載入 CSV 失敗: {e}")
        continue

    # 按 id 排序
    df = df.sort_values('id').reset_index(drop=True)

    # 取出張量內容
    player_ids = tensor_data[:, 0].numpy()
    ids = tensor_data[:, 1].numpy()
    ema_features = tensor_data[:, 2:]
    feature_dim = ema_features.shape[1]

    # 建立 game_id 對應表
    unique_ids = np.unique(ids)
    unique_ids.sort()
    reverse_game_id_mapping = {id_: idx for idx, id_ in enumerate(unique_ids)}

    num_games = len(df)
    num_players_per_game = 10
    tensor_shape = (num_games, num_players_per_game, feature_dim)
    game_ema_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
    game_id_mapping_json = {}

    missing_game_ids = []
    missing_player_ids = []

    for i, row in df.iterrows():
        game_id = row['id']
        game_id_index = reverse_game_id_mapping.get(game_id, -1)
        game_id_mapping_json[str(i)] = str(int(game_id))

        if game_id_index == -1:
            missing_game_ids.append(game_id)
            game_ema_tensor[i] = torch.zeros((num_players_per_game, feature_dim), dtype=torch.float32)
            print(f"⚠️ Warning: game_id {game_id} not found in tensor data")
            continue

        player_cols = [f'player{j+1}' for j in range(num_players_per_game)]
        game_player_ids = [row[col] if pd.notna(row[col]) and row[col] != -1 else -1 for col in player_cols]
        game_mask = ids == game_id
        game_data = tensor_data[game_mask]

        for j, player_id in enumerate(game_player_ids):
            player_id = float(player_id) if player_id != -1 else -1.0
            if player_id == -1.0:
                missing_player_ids.append((game_id, player_id))
                game_ema_tensor[i, j] = torch.zeros(feature_dim, dtype=torch.float32)
                continue

            player_mask = game_data[:, 0] == player_id
            if player_mask.sum() == 0:
                missing_player_ids.append((game_id, player_id))
                game_ema_tensor[i, j] = torch.zeros(feature_dim, dtype=torch.float32)
                print(f"⚠️ Warning: player_id {player_id} not found in game_id {game_id}")
                continue

            player_features = game_data[player_mask, 2:][0]
            game_ema_tensor[i, j] = player_features

    # 加權處理
    mp_index = 20
    if mp_index >= feature_dim:
        raise ValueError(f"mp_index {mp_index} 超出特徵範圍 (0 to {feature_dim-1})")

    MP_min_ema = game_ema_tensor[:, :, mp_index].unsqueeze(-1)
    weights = (MP_min_ema / 48.0).clip(min=0.0)
    weights = 1
    game_ema_tensor = game_ema_tensor * weights

    # 標準化
    do_standardization = True
    if do_standardization:
        feature_means = game_ema_tensor.view(-1, feature_dim).mean(dim=0)
        feature_stds = game_ema_tensor.view(-1, feature_dim).std(dim=0)
        feature_stds[feature_stds == 0] = 1.0
        game_ema_tensor = (game_ema_tensor - feature_means) / feature_stds

    # 輸出路徑
    out_path = os.path.join(output_dir, f"ematensor_{year_suffix}.pt")
    torch.save(game_ema_tensor, out_path)
    with open(mapping_path, 'w') as f:
        json.dump(game_id_mapping_json, f)

    # 打印結果
    print(f"\n✅ 年份 {year_suffix} 完成儲存")
    print(f"→ 張量已儲存至：{out_path}")
    print(f"→ Mapping 已儲存至：{mapping_path}")
    print(f"→ 張量形狀: {game_ema_tensor.shape}")
    print(f"→ 特徵數: {feature_dim}")
    print(f"→ 缺失 game IDs: {len(missing_game_ids)} / 缺失 player IDs: {len(missing_player_ids)}")

    if missing_game_ids:
        print(f"Sample missing game IDs (up to 5): {missing_game_ids[:5]}")
    if missing_player_ids:
        print(f"Sample missing player IDs (up to 5): {missing_player_ids[:5]}")

    # 顯示 player_id 和 game_id 範圍
    unique_player_ids = np.unique([pid for pid in player_ids if pid != -1])
    print(f"Player ID 範圍: [{np.min(unique_player_ids):.0f}, {np.max(unique_player_ids):.0f}]")
    print(f"Game ID 範圍 (連續索引): [0, {num_games-1}]")
    print(f"原始 Game ID 範圍: [{np.min(unique_ids):.0f}, {np.max(unique_ids):.0f}]")

    # 隨機打印一筆資料
    random_game_idx = random.randint(0, num_games-1)
    random_player_idx = random.randint(0, num_players_per_game-1)
    while game_ema_tensor[random_game_idx, random_player_idx].sum() == 0:
        random_game_idx = random.randint(0, num_games-1)
        random_player_idx = random.randint(0, num_players_per_game-1)

    random_features = game_ema_tensor[random_game_idx, random_player_idx].numpy()
    random_game_id = game_id_mapping_json.get(str(random_game_idx), "Unknown")
    random_player_id = df.iloc[random_game_idx][f'player{random_player_idx+1}']

    print(f"\n✅ 隨機資料範例（年份 {year_suffix}）:")
    print(f"Game ID (連續索引): {random_game_idx}, 原始 Game ID: {random_game_id}")
    print(f"Player ID: {random_player_id:.0f}")
    print(f"EMA Features (標準化後):")
    for i, feat in enumerate(random_features):
        print(f"  Feature {i}: {feat:.4f}")

