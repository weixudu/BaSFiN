import csv
import re

def parse_playoff_data(data):
    series_name = ""
    games_data = []
    series_re = re.compile(r'^(.*?Round|.*?Semifinals|.*?Finals)')

    for line in data.split('\n'):
        line = line.strip()
        print(f"Processing line: {line}")
        
        series_match = series_re.match(line)
        if series_match:
            series_name = series_match.group(1).strip()
            print(f"Found series: {series_name}")
        elif line.startswith('Game'):
            parts = line.split()
            game_th = parts[1]
            date = f"{parts[3]}/{parts[4]}"
            
            # 提取隊伍名稱和比分
            at_index = parts.index('@')
            score_a_index = at_index - 1
            score_b_index = len(parts) - 1

            team_a = ' '.join(parts[5:score_a_index])
            score_a = parts[score_a_index]
            team_b = ' '.join(parts[at_index + 1:score_b_index])
            score_b = parts[score_b_index]
            host_team = team_b  # 主場隊伍為@後面的隊伍

            games_data.append([series_name, game_th, date, team_a, score_a, team_b, score_b, host_team])
            print(f"Added game: {games_data[-1]}")

    return games_data

def write_to_csv(games_data, filename='playoff_results.csv'):
    headers = ['series_name', 'game_th', 'date', 'team_a', 'score_a', 'team_b', 'score_b', 'host_team']

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(games_data)

    print(f"Data written to {filename}")
if __name__ == "__main__":
    playoff_data = """
Playoff Series 
Finals	Golden State Warriors over Cleveland Cavaliers  (4-0)	Series Stats
Game 1	Thu, May 31	Cleveland Cavaliers	114	@ Golden State Warriors	124
Game 2	Sun, June 3	Cleveland Cavaliers	103	@ Golden State Warriors	122
Game 3	Wed, June 6	Golden State Warriors	110	@ Cleveland Cavaliers	102
Game 4	Fri, June 8	Golden State Warriors	108	@ Cleveland Cavaliers	85
Eastern Conference Finals	Cleveland Cavaliers over Boston Celtics  (4-3)	Series Stats
Game 1	Sun, May 13	Cleveland Cavaliers	83	@ Boston Celtics	108
Game 2	Tue, May 15	Cleveland Cavaliers	94	@ Boston Celtics	107
Game 3	Sat, May 19	Boston Celtics	86	@ Cleveland Cavaliers	116
Game 4	Mon, May 21	Boston Celtics	102	@ Cleveland Cavaliers	111
Game 5	Wed, May 23	Cleveland Cavaliers	83	@ Boston Celtics	96
Game 6	Fri, May 25	Boston Celtics	99	@ Cleveland Cavaliers	109
Game 7	Sun, May 27	Cleveland Cavaliers	87	@ Boston Celtics	79
Western Conference Finals	Golden State Warriors over Houston Rockets  (4-3)	Series Stats
Game 1	Mon, May 14	Golden State Warriors	119	@ Houston Rockets	106
Game 2	Wed, May 16	Golden State Warriors	105	@ Houston Rockets	127
Game 3	Sun, May 20	Houston Rockets	85	@ Golden State Warriors	126
Game 4	Tue, May 22	Houston Rockets	95	@ Golden State Warriors	92
Game 5	Thu, May 24	Golden State Warriors	94	@ Houston Rockets	98
Game 6	Sat, May 26	Houston Rockets	86	@ Golden State Warriors	115
Game 7	Mon, May 28	Golden State Warriors	101	@ Houston Rockets	92
Eastern Conference Semifinals	Boston Celtics over Philadelphia 76ers  (4-1)	Series Stats
Game 1	Mon, April 30	Philadelphia 76ers	101	@ Boston Celtics	117
Game 2	Thu, May 3	Philadelphia 76ers	103	@ Boston Celtics	108
Game 3	Sat, May 5	Boston Celtics	101	@ Philadelphia 76ers	98
Game 4	Mon, May 7	Boston Celtics	92	@ Philadelphia 76ers	103
Game 5	Wed, May 9	Philadelphia 76ers	112	@ Boston Celtics	114
Eastern Conference Semifinals	Cleveland Cavaliers over Toronto Raptors  (4-0)	Series Stats
Game 1	Tue, May 1	Cleveland Cavaliers	113	@ Toronto Raptors	112
Game 2	Thu, May 3	Cleveland Cavaliers	128	@ Toronto Raptors	110
Game 3	Sat, May 5	Toronto Raptors	103	@ Cleveland Cavaliers	105
Game 4	Mon, May 7	Toronto Raptors	93	@ Cleveland Cavaliers	128
Western Conference Semifinals	Golden State Warriors over New Orleans Pelicans  (4-1)	Series Stats
Game 1	Sat, April 28	New Orleans Pelicans	101	@ Golden State Warriors	123
Game 2	Tue, May 1	New Orleans Pelicans	116	@ Golden State Warriors	121
Game 3	Fri, May 4	Golden State Warriors	100	@ New Orleans Pelicans	119
Game 4	Sun, May 6	Golden State Warriors	118	@ New Orleans Pelicans	92
Game 5	Tue, May 8	New Orleans Pelicans	104	@ Golden State Warriors	113
Western Conference Semifinals	Houston Rockets over Utah Jazz  (4-1)	Series Stats
Game 1	Sun, April 29	Utah Jazz	96	@ Houston Rockets	110
Game 2	Wed, May 2	Utah Jazz	116	@ Houston Rockets	108
Game 3	Fri, May 4	Houston Rockets	113	@ Utah Jazz	92
Game 4	Sun, May 6	Houston Rockets	100	@ Utah Jazz	87
Game 5	Tue, May 8	Utah Jazz	102	@ Houston Rockets	112
Eastern Conference First Round	Boston Celtics over Milwaukee Bucks  (4-3)	Series Stats
Game 1	Sun, April 15	Milwaukee Bucks	107	@ Boston Celtics	113
Game 2	Tue, April 17	Milwaukee Bucks	106	@ Boston Celtics	120
Game 3	Fri, April 20	Boston Celtics	92	@ Milwaukee Bucks	116
Game 4	Sun, April 22	Boston Celtics	102	@ Milwaukee Bucks	104
Game 5	Tue, April 24	Milwaukee Bucks	87	@ Boston Celtics	92
Game 6	Thu, April 26	Boston Celtics	86	@ Milwaukee Bucks	97
Game 7	Sat, April 28	Milwaukee Bucks	96	@ Boston Celtics	112
Eastern Conference First Round	Cleveland Cavaliers over Indiana Pacers  (4-3)	Series Stats
Game 1	Sun, April 15	Indiana Pacers	98	@ Cleveland Cavaliers	80
Game 2	Wed, April 18	Indiana Pacers	97	@ Cleveland Cavaliers	100
Game 3	Fri, April 20	Cleveland Cavaliers	90	@ Indiana Pacers	92
Game 4	Sun, April 22	Cleveland Cavaliers	104	@ Indiana Pacers	100
Game 5	Wed, April 25	Indiana Pacers	95	@ Cleveland Cavaliers	98
Game 6	Fri, April 27	Cleveland Cavaliers	87	@ Indiana Pacers	121
Game 7	Sun, April 29	Indiana Pacers	101	@ Cleveland Cavaliers	105
Eastern Conference First Round	Philadelphia 76ers over Miami Heat  (4-1)	Series Stats
Game 1	Sat, April 14	Miami Heat	103	@ Philadelphia 76ers	130
Game 2	Mon, April 16	Miami Heat	113	@ Philadelphia 76ers	103
Game 3	Thu, April 19	Philadelphia 76ers	128	@ Miami Heat	108
Game 4	Sat, April 21	Philadelphia 76ers	106	@ Miami Heat	102
Game 5	Tue, April 24	Miami Heat	91	@ Philadelphia 76ers	104
Eastern Conference First Round	Toronto Raptors over Washington Wizards  (4-2)	Series Stats
Game 1	Sat, April 14	Washington Wizards	106	@ Toronto Raptors	114
Game 2	Tue, April 17	Washington Wizards	119	@ Toronto Raptors	130
Game 3	Fri, April 20	Toronto Raptors	103	@ Washington Wizards	122
Game 4	Sun, April 22	Toronto Raptors	98	@ Washington Wizards	106
Game 5	Wed, April 25	Washington Wizards	98	@ Toronto Raptors	108
Game 6	Fri, April 27	Toronto Raptors	102	@ Washington Wizards	92
Western Conference First Round	Golden State Warriors over San Antonio Spurs  (4-1)	Series Stats
Game 1	Sat, April 14	San Antonio Spurs	92	@ Golden State Warriors	113
Game 2	Mon, April 16	San Antonio Spurs	101	@ Golden State Warriors	116
Game 3	Thu, April 19	Golden State Warriors	110	@ San Antonio Spurs	97
Game 4	Sun, April 22	Golden State Warriors	90	@ San Antonio Spurs	103
Game 5	Tue, April 24	San Antonio Spurs	91	@ Golden State Warriors	99
Western Conference First Round	Houston Rockets over Minnesota Timberwolves  (4-1)	Series Stats
Game 1	Sun, April 15	Minnesota Timberwolves	101	@ Houston Rockets	104
Game 2	Wed, April 18	Minnesota Timberwolves	82	@ Houston Rockets	102
Game 3	Sat, April 21	Houston Rockets	105	@ Minnesota Timberwolves	121
Game 4	Mon, April 23	Houston Rockets	119	@ Minnesota Timberwolves	100
Game 5	Wed, April 25	Minnesota Timberwolves	104	@ Houston Rockets	122
Western Conference First Round	New Orleans Pelicans over Portland Trail Blazers  (4-0)	Series Stats
Game 1	Sat, April 14	New Orleans Pelicans	97	@ Portland Trail Blazers	95
Game 2	Tue, April 17	New Orleans Pelicans	111	@ Portland Trail Blazers	102
Game 3	Thu, April 19	Portland Trail Blazers	102	@ New Orleans Pelicans	119
Game 4	Sat, April 21	Portland Trail Blazers	123	@ New Orleans Pelicans	131
Western Conference First Round	Utah Jazz over Oklahoma City Thunder  (4-2)	Series Stats
Game 1	Sun, April 15	Utah Jazz	108	@ Oklahoma City Thunder	116
Game 2	Wed, April 18	Utah Jazz	102	@ Oklahoma City Thunder	95
Game 3	Sat, April 21	Oklahoma City Thunder	102	@ Utah Jazz	115
Game 4	Mon, April 23	Oklahoma City Thunder	96	@ Utah Jazz	113
Game 5	Wed, April 25	Utah Jazz	99	@ Oklahoma City Thunder	107
Game 6	Fri, April 27	Oklahoma City Thunder	91	@ Utah Jazz	96
    """

    games_data = parse_playoff_data(playoff_data)
    write_to_csv(games_data)
