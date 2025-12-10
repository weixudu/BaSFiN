import requests
from bs4 import BeautifulSoup
import time
import csv


base_url = "https://www.basketball-reference.com/leagues/NBA_{}.html"


years = range(2018, 2025)


output_file = "nba_team_ratings_2018_to_2024.csv"
fields = ["Year", "Team", "Ortg", "Drtg", "Nrtg"]

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fields)

    
    for year in years:
        url = base_url.format(year)
        print(f"正在爬取 {year} 年的數據...")

     
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('div', {'id': 'div_advanced-team'}).find('tbody')
        for row in table.find_all('tr'):
            # 確保不是表格的空行
            if row.find('th', {'scope': 'row'}) is not None:
                team = row.find('td', {'data-stat': 'team'}).text
                ortg = row.find('td', {'data-stat': 'off_rtg'}).text
                drtg = row.find('td', {'data-stat': 'def_rtg'}).text
                nrtg = row.find('td', {'data-stat': 'net_rtg'}).text

                
                writer.writerow([year, team, ortg, drtg, nrtg])

        
        time.sleep(3)

print(f"數據已存入 {output_file}")
