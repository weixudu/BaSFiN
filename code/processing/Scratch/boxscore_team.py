import pandas as pd


df = pd.read_csv('data/boxscore/boxscore_advance.csv')
team_data = []
game_ids = df['id'].unique()


for game_id in game_ids:

    game_df = df[df['id'] == game_id]
    teams = game_df['team'].unique()
    
    for team in teams:
        
        team_df = game_df[game_df['team'] == team]
        total_mp = team_df['MP'].sum()
        
        
        weighted_ortg = (team_df['ORtg'] * team_df['MP']).sum() / total_mp
        weighted_drtg = (team_df['DRtg'] * team_df['MP']).sum() / total_mp
        weighted_nrtg = weighted_ortg - weighted_drtg
        
        
        team_data.append({
            'id': game_id,
            'team': team,
            'ORtg': weighted_ortg,
            'DRtg': weighted_drtg,
            'Nrtg': weighted_nrtg
        })


team_df_result = pd.DataFrame(team_data)
team_df_result.to_csv('box_team.csv', index=False)

print("結果已保存至 box_team.csv")
