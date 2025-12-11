import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import random
import csv
from tqdm import tqdm

# 球隊全名到縮寫的映射
team_abbr = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "New Jersey Nets": "NJN",
    "Charlotte Hornets": "CHO",
    "Charlotte Bobcats":"CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New Orleans Hornets" : "NOH",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}



def make_request(url, headers, max_retries=5):
    for attempt in range(max_retries):
        try:
            time.sleep(3)  # 遵守 Crawl-delay，設置為 3 秒
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"遇到 429 錯誤，等待 {wait_time:.2f} 秒後重試...")
                time.sleep(wait_time)
            else:
                print(f"請求錯誤: {e}")
                if attempt == max_retries - 1:
                    raise

def scrape_games():
    months_full = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
    months_2012 = ['december', 'january', 'february', 'march', 'april', 'may', 'june']

    years = list(range(2014,2019))  
    
    headers = {
        'User-Agent': 'YourBot/1.0 (+http://www.yourwebsite.com/bot.html)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    game_counter = 6327  # 從1開始計數
    
    for year in years:
        months = months_2012 if year == 2012 else months_full
        game_data = []  # 每年初始化為空數據


        for month in months:
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
            response = make_request(url, headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'schedule'})
            
            if not table:
                print(f"No data found for {year}-{month}.")
                continue

            rows = table.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 4:
                    continue

                date_link = row.find('th', {'data-stat': 'date_game'}).find('a')
                if date_link:
                    date_raw = date_link.text.strip()
                    date_obj = datetime.strptime(date_raw, '%a, %b %d, %Y')
                    date = date_obj.strftime('%b %d')

                    visitor = cols[1].text.strip()
                    visitor_pts = cols[2].text.strip()
                    home = cols[3].text.strip()
                    home_pts = cols[4].text.strip()

                    boxscore_link = row.find('a', string='Box Score')
                    if boxscore_link:
                        boxscore_url = f"https://www.basketball-reference.com{boxscore_link['href']}"
                        print(f"Game ID: {game_counter}")
                        print(f"Date: {date}")
                        print(f"Teams: {visitor} ({visitor_pts}) vs {home} ({home_pts})")
                        print(f"Box Score URL: {boxscore_url}")

                        boxscore_response = make_request(boxscore_url, headers)
                        boxscore_soup = BeautifulSoup(boxscore_response.content, 'html.parser')

                        for team in [visitor, home]:
                            team_abbr_code = team_abbr.get(team, team[:3].upper())  # 使用縮寫，如果沒有則使用前三個字母
                            team_box = boxscore_soup.find('div', {'id': f'div_box-{team_abbr_code}-game-advanced'})
                            if not team_box:
                                print(f"No box score data found for {team}")
                                continue

                            players = team_box.find_all('tr')
                            starters = players[2:7]  # 只選擇前五個球員

                            for player in starters:
                                player_data = player.find_all('td')
                                
                                player_name_element = player.find('th', {'data-stat': 'player'})
                                
                                if player_name_element:
                                    player_name = player_name_element.text.strip()
                                else:
                                    print("Player name not found, skipping player.")
                                    continue
                                
                                mp_raw = player_data[0].text.strip() if len(player_data) > 0 else "0"
                                if ':' in mp_raw:
                                    mp_min = mp_raw.split(':')[0]
                                else:
                                    mp_min = mp_raw

                                data = {
                                    'game_id': game_counter,
                                    'team': team_abbr_code,
                                    'player_name': player_name,
                                    'MP': mp_min,
                                    'TS%': player_data[1].text.strip() if len(player_data) > 1 else "0",
                                    'eFG%': player_data[2].text.strip() if len(player_data) > 2 else "0",
                                    '3PAr': player_data[3].text.strip() if len(player_data) > 3 else "0",
                                    'FTr': player_data[4].text.strip() if len(player_data) > 4 else "0",
                                    'ORB%': player_data[5].text.strip() if len(player_data) > 5 else "0",
                                    'DRB%': player_data[6].text.strip() if len(player_data) > 6 else "0",
                                    'TRB%': player_data[7].text.strip() if len(player_data) > 7 else "0",
                                    'AST%': player_data[8].text.strip() if len(player_data) > 8 else "0",
                                    'STL%': player_data[9].text.strip() if len(player_data) > 9 else "0",
                                    'BLK%': player_data[10].text.strip() if len(player_data) > 10 else "0",
                                    'TOV%': player_data[11].text.strip() if len(player_data) > 11 else "0",
                                    'USG%': player_data[12].text.strip() if len(player_data) > 12 else "0",
                                    'ORtg': player_data[13].text.strip() if len(player_data) > 13 else "0",
                                    'DRtg': player_data[14].text.strip() if len(player_data) > 14 else "0",
                                    'BPM': player_data[15].text.strip() if len(player_data) > 15 else "0"
                                }
                                
                                game_data.append(data)

                        game_counter += 1  # 增加比賽計數器

 
        
        # 儲存當年資料為 CSV 文件
        with open(f"boxscore_advance_{year}.csv", 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'game_id', 'team', 'player_name', 'MP', 'TS%', 'eFG%', '3PAr', 'FTr', 
                'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 
                'ORtg', 'DRtg', 'BPM'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in game_data:
                writer.writerow(data)



scrape_games()