# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# def get_html(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.text
#     else:
#         print(f"Failed to retrieve URL: {url}")
#         return None

# def get_advanced_stats(season):
#     url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
#     html = get_html(url)
#     if not html:
#         return None

#     soup = BeautifulSoup(html, 'html.parser')

#     stats = []
#     table = soup.find('table', id='advanced-team')
#     if table:
#         print(f"Fetching advanced stats for season {season}...")
#         for row in table.tbody.find_all('tr'):
#             rk = row.find('th', attrs={'data-stat': 'ranker'}).text.strip()
#             team = row.find('td', attrs={'data-stat': 'team'}).text.strip().rstrip('*')  # Remove asterisk
#             age = row.find('td', attrs={'data-stat': 'age'}).text.strip()
#             wins = row.find('td', attrs={'data-stat': 'wins'}).text.strip()
#             losses = row.find('td', attrs={'data-stat': 'losses'}).text.strip()
#             pw = row.find('td', attrs={'data-stat': 'wins_pyth'}).text.strip()
#             pl = row.find('td', attrs={'data-stat': 'losses_pyth'}).text.strip()
#             mov = row.find('td', attrs={'data-stat': 'mov'}).text.strip()
#             sos = row.find('td', attrs={'data-stat': 'sos'}).text.strip()
#             srs = row.find('td', attrs={'data-stat': 'srs'}).text.strip()
#             ortg = row.find('td', attrs={'data-stat': 'off_rtg'}).text.strip()
#             drtg = row.find('td', attrs={'data-stat': 'def_rtg'}).text.strip()
#             nrtg = row.find('td', attrs={'data-stat': 'net_rtg'}).text.strip()
#             pace = row.find('td', attrs={'data-stat': 'pace'}).text.strip()
#             ftr = row.find('td', attrs={'data-stat': 'fta_per_fga_pct'}).text.strip()
#             three_par = row.find('td', attrs={'data-stat': 'fg3a_per_fga_pct'}).text.strip()
#             ts_pct = row.find('td', attrs={'data-stat': 'ts_pct'}).text.strip()
#             efg_pct = row.find('td', attrs={'data-stat': 'efg_pct'}).text.strip()
#             tov_pct = row.find('td', attrs={'data-stat': 'tov_pct'}).text.strip()
#             orb_pct = row.find('td', attrs={'data-stat': 'orb_pct'}).text.strip()
#             ft_fga = row.find('td', attrs={'data-stat': 'ft_rate'}).text.strip()
#             drb_pct = row.find('td', attrs={'data-stat': 'drb_pct'}).text.strip()
#             arena = row.find('td', attrs={'data-stat': 'arena_name'}).text.strip()
#             attendance = row.find('td', attrs={'data-stat': 'attendance'}).text.strip()
#             attendance_per_game = row.find('td', attrs={'data-stat': 'attendance_per_g'}).text.strip()

#             stats.append([season, rk, team, age, wins, losses, pw, pl, mov, sos, srs, ortg, drtg, nrtg, pace, ftr,
#                           three_par, ts_pct, efg_pct, tov_pct, orb_pct, ft_fga, drb_pct, arena, attendance,
#                           attendance_per_game])
    
#     return stats


# seasons = list(range(2018, 2025))
# all_stats = []

# for season in seasons:
#     print(f"Fetching advanced stats for season {season}...")
#     stats = get_advanced_stats(season)
#     if stats:
#         all_stats.extend(stats)
#     time.sleep(1)  # Avoid being blocked by the website

# # Save advanced stats data to CSV file
# stats_columns = ["Season", "Rk", "Team", "Age", "W", "L", "PW", "PL", "MOV", "SOS", "SRS", "ORtg", "DRtg", "NRtg",
#                  "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", "FT/FGA", "DRB%", "Arena", "Attendance", "Attendance/G"]
# stats_df = pd.DataFrame(all_stats, columns=stats_columns)
# stats_df.to_csv('advanced_stats.csv', index=False)

# print("Advanced stats data saved to advanced_stats.csv.")
