import requests
import csv

# 獲取比賽列表
response = requests.get('https://codeforces.com/api/contest.list')
if response.status_code == 200:
    data = response.json()
    if 'result' in data:
        contests = data['result']
        print("比賽列表:", contests)
    else:
        print("Error: 'result' key not found in response")
        print("Full response:", data)
else:
    print(f"Failed to retrieve contest list, status code: {response.status_code}")

# 檢查是否有獲取到比賽
if 'result' in data:
    # 取出第一個比賽的ID
    contest_id = contests[0]['id']

    # 獲取比賽排名
    response = requests.get(f'https://codeforces.com/api/contest.standings?contestId={contest_id}&from=1&count=10')
    if response.status_code == 200:
        data = response.json()
        if 'result' in data:
            standings = data['result']['rows']
            print("比賽排名:", standings)

            # 保存比賽排名到CSV文件
            with open('contest_standings.csv', mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Rank', 'Handle', 'Points', 'Penalty'])

                for row in standings:
                    rank = row['rank']
                    handle = row['party']['members'][0]['handle']
                    points = row['points']
                    penalty = row['penalty']
                    writer.writerow([rank, handle, points, penalty])

            print("比賽排名已保存到 contest_standings.csv")

        else:
            print("Error: 'result' key not found in response")
            print("Full response:", data)
    else:
        print(f"Failed to retrieve contest standings, status code: {response.status_code}")

# 獲取用戶評分
user_handle = 'tourist'
response = requests.get(f'https://codeforces.com/api/user.rating?handle={user_handle}')
if response.status_code == 200:
    data = response.json()
    if 'result' in data:
        ratings = data['result']
        print("用戶評分:", ratings)

        # 保存用戶評分到CSV文件
        with open('user_ratings.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ContestID', 'ContestName', 'OldRating', 'NewRating', 'Rank', 'RatingUpdateTimeSeconds'])

            for rating in ratings:
                contest_id = rating['contestId']
                contest_name = rating['contestName']
                old_rating = rating['oldRating']
                new_rating = rating['newRating']
                rank = rating['rank']
                rating_update_time = rating['ratingUpdateTimeSeconds']
                writer.writerow([contest_id, contest_name, old_rating, new_rating, rank, rating_update_time])

        print("用戶評分已保存到 user_ratings.csv")

    else:
        print("Error: 'result' key not found in response")
        print("Full response:", data)
else:
    print(f"Failed to retrieve user rating, status code: {response.status_code}")
