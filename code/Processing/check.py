import pandas as pd
import random

# Team abbreviation mapping
team_abbr = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK", "New Jersey Nets": "NJN",
    "Charlotte Hornets": "CHO", "Charlotte Bobcats": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New Orleans Hornets": "NOH", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# Load data
output_csv = '../data/year_merge_output/data_2009_2015.csv'
input_csv = '../data/year_merge/data_merge_2009_2015.csv'
player_mapping_csv = 'player_id_mapping_2009_2024.csv'
game_result_csv = '../data/game_result/nba_games_2009_2024.csv'

output_df = pd.read_csv(output_csv)
input_df = pd.read_csv(input_csv)
player_df = pd.read_csv(player_mapping_csv)
game_result_df = pd.read_csv(game_result_csv)

# Map team names to abbreviations
game_result_df['Home_Abbr'] = game_result_df['Home'].map(team_abbr)
game_result_df['Visitor_Abbr'] = game_result_df['Visitor'].map(team_abbr)

# Sample 5 games
sample_ids = random.sample(list(output_df['id'].unique()), min(5, len(output_df['id'].unique())))

for game_id in sample_ids:
    print(f"\n🔍 Checking Game ID: {game_id}")
    game_data = output_df[output_df['id'] == game_id].iloc[0]
    game_result = game_result_df[game_result_df['id'] == game_id].iloc[0]
    
    home_team, home_abbr = game_result['Home'], game_result['Home_Abbr']
    visitor_team, visitor_abbr = game_result['Visitor'], game_result['Visitor_Abbr']
    target = game_data['target']
    
    print(f"Home: {home_team} ({home_abbr}), Away: {visitor_team} ({visitor_abbr})")
    print(f"Target: {target} ({'Home Win' if target == 1 else 'Away Win'})")
    
    # Expected player IDs
    home_players = input_df[(input_df['id'] == game_id) & (input_df['team'] == home_abbr)]['player_name']
    away_players = input_df[(input_df['id'] == game_id) & (input_df['team'] == visitor_abbr)]['player_name']
    expected_home_ids = player_df[player_df['player_name'].isin(home_players)]['player_id'].tolist()[:5]
    expected_away_ids = player_df[player_df['player_name'].isin(away_players)]['player_id'].tolist()[:5]
    
    # Actual player IDs
    actual_home_ids = game_data[[f'player{i+1}' for i in range(5)]].dropna().tolist()
    actual_away_ids = game_data[[f'player{i+6}' for i in range(5)]].dropna().tolist()
    
    # Check home players
    print("\nHome Players (player1 to player5):")
    print(f"Expected: {expected_home_ids}")
    print(f"Actual: {actual_home_ids}")
    home_correct = all(pid in expected_home_ids for pid in actual_home_ids) and len(actual_home_ids) <= 5
    print("✅ Home Players Correct" if home_correct else "❌ Home Players Incorrect")
    
    # Check away players
    print("\nAway Players (player6 to player10):")
    print(f"Expected: {expected_away_ids}")
    print(f"Actual: {actual_away_ids}")
    away_correct = all(pid in expected_away_ids for pid in actual_away_ids) and len(actual_away_ids) <= 5
    print("✅ Away Players Correct" if away_correct else "❌ Away Players Incorrect")
    
    # Check target
    expected_target = game_result['target']
    print(f"\nTarget Check:")
    print(f"Expected: {expected_target}")
    print(f"Actual: {target}")
    print("✅ Target Correct" if target == expected_target else "❌ Target Incorrect")