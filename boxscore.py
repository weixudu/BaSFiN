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

    years = list(range(2009,2015))   
    headers = {
        'User-Agent': 'YourBot/1.0 (+http://www.yourwebsite.com/bot.html)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    game_counter = 1  
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
                    game_id = game_counter

                    visitor = cols[1].text.strip()
                    visitor_pts = cols[2].text.strip()
                    home = cols[3].text.strip()
                    home_pts = cols[4].text.strip()

                    boxscore_link = row.find('a', string='Box Score')
                    if boxscore_link:
                        boxscore_url = f"https://www.basketball-reference.com{boxscore_link['href']}"
                        print(f"Game ID: {game_id}")
                        #print(f"Date: {date}")
                        teama=team_abbr.get(visitor)
                        teamb =team_abbr.get(home)
                        print(f"Teams: {teama} ({visitor_pts}) vs {teamb} ({home_pts})")
                        #print(f"Box Score URL: {boxscore_url}")

                        boxscore_response = make_request(boxscore_url, headers)
                        boxscore_soup = BeautifulSoup(boxscore_response.content, 'html.parser')

                        for team in [visitor, home]:
                            team_abbr_code = team_abbr.get(team, team[:3].upper())  # 使用縮寫，如果沒有則使用前三個字母
                            team_box = boxscore_soup.find('div', {'id': f'div_box-{team_abbr_code}-game-basic'})
                            if not team_box:
                                print(f"No box score data found for {team}")
                                continue

                            players = team_box.find_all('tr')
                            starters = players[2:7]  # 只選擇前五個球員

                            for player in starters:
                                player_data = player.find_all('td')
                                if len(player_data) < 20:
                                    continue
                                player_name = player.find('th').text.strip()

                                mp_raw = player_data[0].text.strip()
                                if ':' in mp_raw:
                                    mp_min = mp_raw.split(':')[0]  # 只取分鐘數
                                else:
                                    mp_min = mp_raw  # 如果沒有':'，直接視為分鐘數

                                data = {
                                    'game_id': game_id,
                                    'team': team_abbr_code,
                                    'player_name': player_name,
                                    'MP_min': mp_min,
                                    'FG': player_data[1].text.strip(),
                                    'FGA': player_data[2].text.strip(),
                                    'FG%': player_data[3].text.strip(),
                                    '3P': player_data[4].text.strip(),
                                    '3PA': player_data[5].text.strip(),
                                    '3P%': player_data[6].text.strip(),
                                    'FT': player_data[7].text.strip(),
                                    'FTA': player_data[8].text.strip(),
                                    'FT%': player_data[9].text.strip(),
                                    'ORB': player_data[10].text.strip(),
                                    'DRB': player_data[11].text.strip(),
                                    'TRB': player_data[12].text.strip(),
                                    'AST': player_data[13].text.strip(),
                                    'STL': player_data[14].text.strip(),
                                    'BLK': player_data[15].text.strip(),
                                    'TOV': player_data[16].text.strip(),
                                    'PF': player_data[17].text.strip(),
                                    'PTS': player_data[18].text.strip(),
                                    'GmSc': player_data[19].text.strip(),
                                    '+/-': player_data[20].text.strip() if len(player_data) > 20 else "N/A"
                                }
                                game_data.append(data)

                        game_counter += 1  # 增加比賽計數器

        # 儲存為年度 CSV 文件
        csv_filename = f'boxscore_{year}.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['game_id', 'team', 'player_name', 'MP_min', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in game_data:
                writer.writerow(data)

        print(f"儲存完成: {csv_filename}")

scrape_games()
