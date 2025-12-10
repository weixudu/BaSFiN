import pandas as pd

df = pd.read_csv("../data/final_data/data_2013_2024.csv")   


# ------------- 函式：鎖定只在 2023 / 2024 出現的玩家 -------------
def players_only_in_23_or_24(df: pd.DataFrame):
    """
    回傳：
        - all_23_24 : 同時包含「只在 2023」與「只在 2024」的 player 列表
        - only_2023 : 只在 2023 出現的 player 列表
        - only_2024 : 只在 2024 出現的 player 列表
    """
    # 1) 找出所有 player 欄位（player1 ~ player10）
    player_cols = [c for c in df.columns if c.lower().startswith("player")]
    
    # 2) 轉成 long format：每列為 (year, player_id)
    long_df = (
        df.melt(id_vars="year", value_vars=player_cols,
                value_name="player_id")      # 將 playerX 攤平成一欄
          .dropna(subset=["player_id"])       # 跳過空值
          .astype({"player_id": int})         # 型別一致
    )
    
    # 3) 整理每位 player 出現過的年份集合
    years_by_player = long_df.groupby("player_id")["year"].agg(set)
    
    # 4) 篩出「出現年份集合 ⊆ {2023, 2024}」者
    target_years = {2023, 2024}
    all_23_24 = [pid for pid, yrs in years_by_player.items() if yrs.issubset(target_years)]
    
    # 進一步拆成「只 2023」與「只 2024」
    only_2023 = [pid for pid, yrs in years_by_player.items() if yrs == {2023}]
    only_2024 = [pid for pid, yrs in years_by_player.items() if yrs == {2024}]
    
    return all_23_24, only_2023, only_2024


# ------------------ 使用範例 ------------------
# 假設 df 是你的資料表
all_players, only_2023, only_2024 = players_only_in_23_or_24(df)

print("只在 2023 或 2024 出現的 最小player：", min(all_players))
print("只在 2023 出現的 player：",       only_2023)
print("只在 2024 出現的 player：",       only_2024)
