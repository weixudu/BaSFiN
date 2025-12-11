import pandas as pd
import os

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

# Generate player ID mapping
def generate_player_id_mapping(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    player_ids = {player: idx + 1 for idx, player in enumerate(df['player_name'].unique())}
    player_df = pd.DataFrame(list(player_ids.items()), columns=['player_name', 'player_id'])
    player_df.to_csv(output_csv, index=False)
    print(f"✅ Player ID mapping saved to: {output_csv}")
    return player_df

# Generate game player IDs with correct home/away separation
def generate_game_player_ids(input_df, player_df, game_result_df):
    # Merge player IDs into input data
    df = input_df.merge(player_df, on='player_name', how='left')
    
    # Map team names to abbreviations in game_result_df
    game_result_df['Home_Abbr'] = game_result_df['Home'].map(team_abbr)
    game_result_df['Visitor_Abbr'] = game_result_df['Visitor'].map(team_abbr)
    id_to_teams = game_result_df.set_index('id')[['Home_Abbr', 'Visitor_Abbr']].to_dict('index')
    
    # Group by game ID and year
    result_list = []
    for (game_id, year), game_df in df.groupby(['id', 'year']):
        teams = id_to_teams.get(game_id)
        if not teams:
            print(f"⚠️ Game ID {game_id} not found in game_result")
            continue
        
        home_team, away_team = teams['Home_Abbr'], teams['Visitor_Abbr']
        
        # Filter home and away players
        home_players = game_df[game_df['team'] == home_team]['player_id'].tolist()[:5]
        away_players = game_df[game_df['team'] == away_team]['player_id'].tolist()[:5]
        
        # Pad with None if fewer than 5 players
        home_players += [None] * (5 - len(home_players))
        away_players += [None] * (5 - len(away_players))
        
        # Assign players to columns
        game_data = {
            'year': year, 'id': game_id,
            **{f'player{i+1}': home_players[i] for i in range(5)},
            **{f'player{i+6}': away_players[i] for i in range(5)}
        }
        result_list.append(game_data)
    
    return pd.DataFrame(result_list)

# Add target from game_result_csv
def add_game_result_target(df_grouped, game_result_csv):
    games_result_df = pd.read_csv(game_result_csv)[['id', 'target']]
    return df_grouped.merge(games_result_df, on='id', how='left')

# Main processing function
def main():
    # File paths
    mapping_input_csv = '../data/boxscore/boxscore_basic_2009_2024.csv'
    game_result_csv = '../data/game_result/nba_games_2009_2024.csv'
    year_merge_dir = '../data/feature_csv'
    output_dir = '../data/final_data'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate player ID mapping
    player_mapping_df = generate_player_id_mapping(mapping_input_csv, 'player_id_mapping_2009_2024.csv')
    game_result_df = pd.read_csv(game_result_csv)
    
    # Process each year_merge file
    for file in os.listdir(year_merge_dir):
        if file.startswith('data_merge') and file.endswith('.csv'):
            input_path = os.path.join(year_merge_dir, file)
            print(f"🔄 Processing: {input_path}")
            
            input_df = pd.read_csv(input_path)
            df_grouped = generate_game_player_ids(input_df, player_mapping_df, game_result_df)
            final_df = add_game_result_target(df_grouped, game_result_csv)
            
            # Ensure only required columns
            cols = ['year', 'id'] + [f'player{i+1}' for i in range(10)] + ['target']
            final_df = final_df[cols]
            
            # Save output
            year_range = file.replace('data_merge_', '').replace('.csv', '')
            output_path = os.path.join(output_dir, f'data_{year_range}.csv')
            final_df.to_csv(output_path, index=False)
            print(f"✅ Saved: {output_path}, {len(final_df)} records\n")

if __name__ == "__main__":
    main()