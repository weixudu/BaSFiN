import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve URL: {url}")
        return None

def get_conference_standings(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
    html = get_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    standings = []
    # Extract Western Conference Standings
    western_conf_standings = soup.find('div', id='div_confs_standings_W')
    if western_conf_standings:
        table = western_conf_standings.find('table')
        if table:
            conf = "Western"
            for row in table.tbody.find_all('tr'):
                team = row.find('a').text
                data = [td.text if td.text != "-" else "0" for td in row.find_all('td')]
                standings.append([season, conf, team] + data)
    
    # Extract Eastern Conference Standings
    eastern_conf_standings = soup.find('div', id='div_confs_standings_E')
    if eastern_conf_standings:
        table = eastern_conf_standings.find('table')
        if table:
            conf = "Eastern"
            for row in table.tbody.find_all('tr'):
                team = row.find('a').text
                data = [td.text if td.text != "-" else "0" for td in row.find_all('td')]
                standings.append([season, conf, team] + data)
    
    return standings

seasons = list(range(2018, 2025))
all_standings = []

for season in seasons:
    print(f"Fetching standings for season {season}...")
    standings = get_conference_standings(season)
    if standings:
        all_standings.extend(standings)
    time.sleep(1)  # Avoid being blocked by the website
#print(all_standings)
# Replace "-" with 0 in the dataframe
for row in all_standings:
    for i in range(len(row)):
        if row[i] == "—":
            row[i] = "0"

# Save standings data to CSV file
standings_columns = ["Season", "Conference", "Team", "W", "L", "W/L%", "GB", "PS/G", "PA/G", "SRS"]
standings_df = pd.DataFrame(all_standings, columns=standings_columns)
standings_df.to_csv('conference_standings.csv', index=False)

print("Conference standings data saved to conference_standings.csv.")
