import pandas as pd

# 檔案路徑
stage2_path = "../output/BaS/player_skills_stage2_kl0.017433288221999882_2025-06-24_12-33-37.csv"
test_path = "../output/BaS/player_stats_2025-06-24_12-33-37.csv"

# 讀取 stage2 資料（包含三個 trial）
df_stage2 = pd.read_csv(stage2_path)

# 平均每位玩家的 Skill Mu 與 Sigma（根據三個 Trial）
df_stage2_avg = (
    df_stage2.groupby("Player_ID")[["Skill_Mu", "Skill_Sigma"]]
    .mean()
    .reset_index()
    .rename(columns={
        "Skill_Mu": "Stage2_Skill_Mu_Avg",
        "Skill_Sigma": "Stage2_Skill_Sigma_Avg"
    })
)

# 讀取測試集結果資料
df_test = pd.read_csv(test_path)

# 合併兩份資料，依照 Player_ID 連接
df_merged = pd.merge(df_stage2_avg, df_test, on="Player_ID", how="inner")

# 最終資料順序整理
df_merged = df_merged[[
    "Player_ID", "Games", "Wins", "Win_Rate",
    "Stage2_Skill_Mu_Avg", "Stage2_Skill_Sigma_Avg"
]]

# 依照 Stage2_Skill_Sigma_Avg 排序
df_sorted = df_merged.sort_values(by="Stage2_Skill_Sigma_Avg")

# 取出 sigma 最低與最高的前10筆資料
lowest_sigma = df_sorted.head(5)
highest_sigma = df_sorted.tail(5)

# 打印結果
print("⭐ Sigma 最低前 5 名：")
print(lowest_sigma.to_string(index=False))

print("\n🚨 Sigma 最高前 5 名：")
print(highest_sigma.to_string(index=False))

# 新增：打印 sigma > 1 的筆數與對應指標資料
sigma_gt_1 = df_merged[df_merged["Stage2_Skill_Sigma_Avg"] > 1.000002]
print(f"\n🔔 Sigma > 1 的筆數: {len(sigma_gt_1)}")
print(f"\n🔔 所有選手: {len(df_merged)}")
print(f"\n🔔 Sigma > 1 的比例: {len(sigma_gt_1)/len(df_merged)}")
print("這些玩家的資料如下：")
print(sigma_gt_1.to_string(index=False))


# ======== 自訂門檻 (分位數) ========
# 場次 > Games 的 50% 分位，且 σ > σ 的 50% 分位
GAMES_Q = 0.5
SIGMA_Q = 0.5

# ------------- 5) 自動尋找「高場次 + 高 σ」選手 (Player_ID < 1221) -------------
df = df_merged
games_thr  = df["Games"].quantile(GAMES_Q)
sigma_thr  = df["Stage2_Skill_Sigma_Avg"].quantile(SIGMA_Q)

high_games_high_sigma = (
    df[
        (df["Games"] >= games_thr) &
        (df["Stage2_Skill_Sigma_Avg"] >= sigma_thr) &
        (df["Player_ID"] < 1221)            # ★ 新增：ID < 1221
    ]
    .sort_values(by="Games", ascending=False)   # 依 Games 由大到小排序
)

print(
    f"\n🎯 高場次 (Games ≥ {games_thr:.0f})、高 σ (σ ≥ {sigma_thr:.3f}) "
    f"且 Player_ID < 1221 的目標選手：{len(high_games_high_sigma)} 位"
)
print(
    high_games_high_sigma[
        ["Player_ID", "Games", "Win_Rate",
         "Stage2_Skill_Mu_Avg", "Stage2_Skill_Sigma_Avg"]
    ].to_string(index=False)
)
