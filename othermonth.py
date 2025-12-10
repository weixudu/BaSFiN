import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

years_months = {
    2020: ['july', 'august', 'september'],
    2021: ['july']
}

all_data = []

for year, months in years_months.items():
    for month in months:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
        
        time.sleep(5)
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data for {year}-{month}, Status Code: {response.status_code}, URL: {url}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'schedule'})

        if not table:
            print(f"No data found for {year}-{month}, URL: {url}")
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

                all_data.append([year, date, visitor, visitor_pts, home, home_pts])

df = pd.DataFrame(all_data, columns=['Season', 'Date', 'Visitor', 'Visitor PTS', 'Home', 'Home PTS'])
df.to_csv('nba_games_2020_2021.csv', index=False)
print("Data has been successfully saved to nba_games_2020_2021.csv")
