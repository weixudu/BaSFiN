import os
import pandas as pd
import numpy as np
import torch
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data_path = "../data/feature_csv/data_merge_2013_2024.csv"
mapping_path = "../data/player_id_mapping_2009_2024.csv"
output_dir = "../data/ema_tensor"

df = pd.read_csv(data_path)
player_mapping = pd.read_csv(mapping_path)

# 映射 player_name → player_id
player_id_dict = dict(zip(player_mapping['player_name'], player_mapping['player_id']))
df['player_id'] = df['player_name'].map(player_id_dict)

if df['player_id'].isna().any():
    print("警告：部分 player_name 無法映射到 player_id，已刪除這些筆數據")
    df = df.dropna(subset=['player_id'])

if "MP" in df.columns:
    df = df.drop(columns=["MP"])

df = df.sort_values(['player_id', 'id']).reset_index(drop=True)
ordered_features = [
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'AST',
    'PTS', 'TS%', 'eFG%', '3PAr', 'FTr', 'AST%', 'USG%', 'ORtg', 'ORB', 'ORB%',
    'DRB', 'DRB%', 'TRB', 'TRB%', 'STL', 'BLK', 'STL%', 'BLK%', 'DRtg',
    'MP_min', 'TOV', 'PF', 'GmSc', '+/-', 'TOV%', 'BPM', 'NRtg'
]

# 「NaN、Inf → 0」並轉成 float64
for col in ordered_features:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float64)

years = sorted(df["year"].unique())   # 例: [2013,2014,...,2024]
train_years = years[:10]   # 2013–2022

# 計算「訓練集整體平均」作為全局的初始 EMA 值（僅用於擬合失敗時）
train_df_allplayers = df[df["year"].isin(train_years)]
global_init_df = train_df_allplayers[ordered_features].mean()
global_init_levels = {
    col: float(global_init_df[col] if not np.isnan(global_init_df[col]) else 0.0)
    for col in ordered_features
}

# 定義：對單一球員、多特徵做 SES 平滑，並一路填充至最後
def compute_player_ets_multi(player_df, cols_to_smooth):
    """
    對單一球員的指定欄位做 SES 平滑，並回傳：
      - full_df: 合併所有年份後的 DataFrame，包含 {col}_ema 與 SES_alpha_{col} 欄位
      - overall_rmse: dict{col: 全量資料 RMSE}
      - alphas: dict{col: SES alpha}
      - fit_failures: dict{col: 0/1，其中 1 表示「跑 SES 失敗」或值全相同等 baseline}
    """
    full_df_sorted = player_df.sort_values('id').reset_index(drop=True)

    alphas = {}
    initial_levels = {}
    overall_rmse = {}
    fit_failures = {col: 0 for col in cols_to_smooth}

    player_id = player_df['player_id'].iloc[0]

    # 用全序列估 alpha 和 initial_level，若失敗則 baseline
    for col in cols_to_smooth:
        series = full_df_sorted[col].astype(np.float64)
        n_points = len(series)
        n_unique = series.nunique()

        # Case 1: 只有 1 筆 → baseline (失敗)
        if n_points <= 1:
            fit_failures[col] = 1
            alphas[col] = 0.05  # 修改為 0.05
            initial_levels[col] = global_init_levels[col]
            overall_rmse[col] = 0.0
            continue

        # Case 2: 值全相同 → baseline (失敗)
        if n_unique <= 1:
            fit_failures[col] = 1
            alphas[col] = 0.05  # 修改為 0.05
            initial_levels[col] = global_init_levels[col]
            # 計算 RMSE
            mse_full = ((series.values - series.values.mean()) ** 2).mean()
            rmse_full = np.sqrt(mse_full) if np.isfinite(mse_full) else 0.0
            overall_rmse[col] = float(rmse_full)
            continue

        # Case 3: 正常：用 statsmodels SES 全量擬合，估 alpha 和 initial_level
        try:
            model = ExponentialSmoothing(
                series,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
                use_boxcox=False
            )
            fit = model.fit(smoothing_level=None, optimized=True, remove_bias=False)
            alpha = fit.params.get("smoothing_level", 0.5)
            initial_level = fit.params.get("initial_level", global_init_levels[col])

            # 用 fittedvalues 跟原始做全量 RMSE
            fitted_vals = fit.fittedvalues.values
            mse_full = ((series.values - fitted_vals) ** 2).mean()
            rmse_full = np.sqrt(mse_full) if np.isfinite(mse_full) else 0.0

        except Exception:
            fit_failures[col] = 1
            alpha = 0.05  # 修改為 0.05
            initial_level = global_init_levels[col]
            # 計算 RMSE
            mse_full = ((series.values - series.values.mean()) ** 2).mean()
            rmse_full = np.sqrt(mse_full) if np.isfinite(mse_full) else 0.0

        # 若結果無效，視為失敗
        if np.isnan(alpha) or np.isnan(rmse_full) or np.isnan(initial_level):
            fit_failures[col] = 1
            alpha = 0.05  # 修改為 0.05
            initial_level = global_init_levels[col]
            mse_full = ((series.values - series.values.mean()) ** 2).mean()
            rmse_full = np.sqrt(mse_full) if np.isfinite(mse_full) else 0.0

        alphas[col] = float(np.clip(alpha, 0.0, 1.0))
        initial_levels[col] = float(initial_level)
        overall_rmse[col] = float(rmse_full)

    # 一路填充：合併所有年份，用同樣 alpha 做因果式 EMA
    full_df = full_df_sorted.copy()

    # 新增 {col}_ema 與 SES_alpha_{col} 欄位
    new_cols = {}
    for col in cols_to_smooth:
        new_cols[f"{col}_ema"] = [np.nan] * len(full_df)
        new_cols[f"SES_alpha_{col}"] = [alphas[col]] * len(full_df)
    full_df = pd.concat([full_df, pd.DataFrame(new_cols, index=full_df.index)], axis=1)

    # 使用 SES 估計的初始水平作為 prev_ema，若失敗則用全局平均值
    prev_ema = {col: initial_levels[col] for col in cols_to_smooth}

    # 逐行做因果式 EMA
    for idx, row in full_df.iterrows():
        for col in cols_to_smooth:
            alpha = alphas[col]
            last_ema = prev_ema[col]
            if idx == 0:
                ema_t = last_ema  # 使用 SES 估計的初始水平
            else:
                prev_y = full_df.at[idx-1, col]
                ema_t = alpha * prev_y + (1 - alpha) * last_ema
            full_df.at[idx, f"{col}_ema"] = ema_t
            prev_ema[col] = ema_t
            full_df.at[idx, f"SES_alpha_{col}"] = alpha

    return full_df, overall_rmse, alphas, fit_failures

# =========================
# 這裡開始替換原本的主程式「處理資料」部分
# =========================

all_data_files = [
    "../data/feature_csv/data_merge_2009_2020.csv",
    "../data/feature_csv/data_merge_2010_2021.csv",
    "../data/feature_csv/data_merge_2011_2022.csv",
    "../data/feature_csv/data_merge_2012_2023.csv",
    "../data/feature_csv/data_merge_2013_2024.csv"
]
for data_path in all_data_files:
    print("\n===================================================")
    print(f"開始處理檔案：{data_path}")
    print("===================================================")

    # 從檔名擷取起始年份
    try:
        fname = os.path.basename(data_path)
        prefix_year = fname.replace('.csv', '').split('_')[3]
    except Exception:
        print(f"❌ 無法從檔名擷取年份，跳過：{data_path}")
        continue

    df = pd.read_csv(data_path)
    player_mapping = pd.read_csv(mapping_path)

    # 映射 player_name → player_id
    player_id_dict = dict(zip(player_mapping['player_name'], player_mapping['player_id']))
    df['player_id'] = df['player_name'].map(player_id_dict)

    if df['player_id'].isna().any():
        print("⚠️  部分 player_name 無法映射到 player_id，已刪除")
        df = df.dropna(subset=['player_id'])

    if "MP" in df.columns:
        df = df.drop(columns=["MP"])

    df = df.sort_values(['player_id', 'id']).reset_index(drop=True)

    # 清理 NaN、Inf
    for col in ordered_features:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float64)

    years = sorted(df["year"].unique())
    if len(years) < 2:
        print("⚠️  年份數量過少，跳過此檔案")
        continue

    train_years = years[:10]
    train_df_allplayers = df[df["year"].isin(train_years)]
    global_init_df = train_df_allplayers[ordered_features].mean()
    global_init_levels = {
        col: float(global_init_df[col] if not np.isnan(global_init_df[col]) else 0.0)
        for col in ordered_features
    }

    print("\n=== 第一階段：估計 SES 參數並篩選特徵 ===")
    all_results_ses = []
    all_overall_rmse_ses = []
    all_alphas_ses = []
    all_failures_ses = []

    for player_id in df["player_id"].unique():
        p_df = df[df["player_id"] == player_id].copy()
        if p_df.empty:
            continue

        result, overall_rmse_scores, alphas, fit_failures = compute_player_ets_multi(
            p_df, ordered_features
        )
        all_results_ses.append(result)
        all_overall_rmse_ses.append(pd.Series(overall_rmse_scores, name=player_id))
        all_alphas_ses.append(pd.Series(alphas, name=player_id))
        all_failures_ses.append(pd.Series(fit_failures, name=player_id))

    first_overall_rmse_df_ses = pd.concat(all_overall_rmse_ses, axis=1).T

    rmse_threshold = 10.0
    high_rmse_features = [
        col for col in ordered_features
        if first_overall_rmse_df_ses[col].mean() >= rmse_threshold
    ]
    cols_to_smooth_ses = [col for col in ordered_features if col not in high_rmse_features]

    print(f"\n→ 篩掉 RMSE >= {rmse_threshold} 的特徵：")
    print(high_rmse_features)
    print(f"→ 最終保留特徵數：{len(cols_to_smooth_ses)}")

    print("\n=== 第二階段：用選定特徵一路做 EMA 並儲存 ===")
    all_results_ses = []
    for player_id in df["player_id"].unique():
        p_df = df[df["player_id"] == player_id].copy()
        if p_df.empty:
            continue

        full_df, _, _, _ = compute_player_ets_multi(p_df, cols_to_smooth_ses)
        all_results_ses.append(full_df)

    final_df_ses = pd.concat(all_results_ses, ignore_index=True)

    numeric_df_ses = final_df_ses.copy()
    numeric_df_ses['player_id'] = numeric_df_ses['player_id'].astype(float)
    numeric_df_ses['id'] = numeric_df_ses['id'].astype(float)
    tensor_cols = ['player_id', 'id'] + [f"{col}_ema" for col in cols_to_smooth_ses]

    tensor_data_ses = torch.tensor(
        numeric_df_ses[tensor_cols].values,
        dtype=torch.float32
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path_ses = os.path.join(output_dir, f"features_ema_rmse{prefix_year}.pt")
    torch.save(tensor_data_ses, out_path_ses)

    print(f"\n✅ 已儲存至：{out_path_ses}")
    print(f"→ 張量維度：{list(tensor_data_ses.shape)}")
