import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

# 定義要抓取的年份
years = range(2009, 2018)

# 月份的設定，除了2012年，其他年份抓取所有月份
months_full = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
months_2012 = ['december', 'january', 'february', 'march', 'april', 'may', 'june']

# 初始化一個空列表來儲存所有資料
all_data = []

for year in years:
    # 根據年份選擇月份，2012 年只從 12 月開始，其他年份抓取所有月份
    months = months_2012 if year == 2012 else months_full
    
    for month in months:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
        
        # 模擬人類行為的延遲，避免被伺服器封鎖
        time.sleep(3)
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data for {year}, Status Code: {response.status_code}, URL: {url}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'schedule'})

        if not table:
            print(f"No data found for {year} {month}, URL: {url}")
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

# 將所有資料轉為DataFrame並儲存為CSV檔案
df = pd.DataFrame(all_data, columns=['Season', 'Date', 'Visitor', 'Visitor PTS', 'Home', 'Home PTS'])
df.to_csv('nba_games_2009_2017.csv', index=False)
print("Data has been successfully saved")
