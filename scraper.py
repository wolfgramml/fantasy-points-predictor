import csv
from selenium import webdriver
from bs4 import BeautifulSoup

team_abbreviations = {
    "GNB": "Green Bay Packers",
    "CHI": "Chicago Bears",
    "KAN": "Kansas City Chiefs",
    "BAL": "Baltimore Ravens",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "ATL": "Atlanta Falcons",
    "ARI": "Arizona Cardinals",
    "BUF": "Buffalo Bills",
    "TEN": "Tennessee Titans",
    "CIN": "Cincinnati Bengals",
    "NWE": "New England Patriots",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "MIA": "Miami Dolphins",
    "JAX": "Jacksonville Jaguars",
    "NOR": "New Orleans Saints",
    "CAR": "Carolina Panthers",
    "MIN": "Minnesota Vikings",
    "NYG": "New York Giants",
    "LVR": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "DEN": "Denver Broncos",
    "SEA": "Seattle Seahawks",
    "DAL": "Dallas Cowboys",
    "CLE": "Cleveland Browns",
    "WAS": "Washington Commanders",
    "TAM": "Tampa Bay Buccaneers",
    "LAR": "Los Angeles Rams",
    "DET": "Detroit Lions",
    "SFO": "San Francisco 49ers",
    "NYJ": "New York Jets",
}

shortened_team_names = {
    "Packers": "Green Bay Packers",
    "Bears": "Chicago Bears",
    "Chiefs": "Kansas City Chiefs",
    "Ravens": "Baltimore Ravens",
    "Eagles": "Philadelphia Eagles",
    "Steelers": "Pittsburgh Steelers",
    "Falcons": "Atlanta Falcons",
    "Cardinals": "Arizona Cardinals",
    "Bills": "Buffalo Bills",
    "Titans": "Tennessee Titans",
    "Bengals": "Cincinnati Bengals",
    "Patriots": "New England Patriots",
    "Texans": "Houston Texans",
    "Colts": "Indianapolis Colts",
    "Dolphins": "Miami Dolphins",
    "Jaguars": "Jacksonville Jaguars",
    "Panthers": "Carolina Panthers",
    "Saints": "New Orleans Saints",
    "Vikings": "Minnesota Vikings",
    "Giants": "New York Giants",
    "Chargers": "Los Angeles Chargers",
    "Raiders": "Las Vegas Raiders",
    "Broncos": "Denver Broncos",
    "Seahawks": "Seattle Seahawks",
    "Cowboys": "Dallas Cowboys",
    "Browns": "Cleveland Browns",
    "Commanders": "Washington Commanders",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Rams": "Los Angeles Rams",
    "Lions": "Detroit Lions",
    "49ers": "San Francisco 49ers",
    "Jets": "New York Jets",
}

url = 'https://www.pro-football-reference.com/boxscores/202409090sfo.htm' # Replace with the URL of the page you want to scrape

driver = webdriver.Chrome()
driver.get(url)

import time
time.sleep(1)
html = driver.page_source

date_str = url.split('/')[4][:8]

# Format date as YYYY-MM-DD
date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

# Step 2: Parse the HTML content
soup = BeautifulSoup(html, 'html.parser')

away_team = ''
home_team = ''
away_score = ''
home_score = ''
start_time = ''

successful_tables = 0

# Step 3: Locate the scorebox and extract team names and scores
scorebox = soup.find('div', class_='scorebox')
if scorebox:
    # Safely extract team names
    team_names = []
    for strong in scorebox.find_all('strong'):
        team_link = strong.find('a')  # Check for the <a> element
        if team_link:
            team_names.append(team_link.text.strip())
    
    away_team = team_names[0]
    home_team = team_names[1]

    # Extract scores
    scores = [int(score.text.strip()) for score in scorebox.find_all('div', class_='score')]

    away_score = scores[0]
    home_score = scores[1]

    start_time_element = scorebox.find('strong', string=lambda text: text and "Start Time" in text)
    start_time_text = start_time_element.next_sibling.strip()
    start_time = start_time_text[2:]

else:
    print("Scorebox not found.")

# content_table = soup.find('div', class_='content_grid')

game_info_div = soup.find('div', id='div_game_info')

# Now locate the table within this specific div
game_info_table = game_info_div.find('table', id='game_info') if game_info_div else None

roof = surface = temperature = windspeed = weather = None

# Extract rows from the table
if game_info_table:
    # game_info_table = content_table.find('table', id='game_info')
    rows = game_info_table.find_all('tr')  # Find all rows in the table
    for row in rows:
        header = row.find('th', {'data-stat': 'info'})  # Extract the row header
        value = row.find('td', {'data-stat': 'stat'})   # Extract the corresponding value
        
        if header and value:
            header_text = header.text.strip()
            value_text = value.text.strip()
            
            # Match specific headers to extract data
            if header_text == "Roof":
                roof = value_text
            elif header_text == "Surface":
                surface = value_text
            elif header_text == "Weather":
                weather = value_text
    if weather:
        temperature = weather.split(" degrees")[0]  # Get the part before " degrees"
        wind_part = weather.split("wind ")[1]  # Get the part after "wind "
        windspeed = wind_part.split(" mph")[0]
    else:
        temperature = windspeed = 'N/A'

else:
    print("Game info table not found.")

away_rushing_yards = home_rushing_yards = away_passing_yards = home_passing_yards = away_total_yards = home_total_yards = None

yards_table = soup.find('table', id='team_stats')

if yards_table:
    rows = yards_table.find_all('tr')
    for row in rows:
        stat_name = row.find('th')  # Find the header (stat name) for each row
        if stat_name:
            stat_name = stat_name.text.strip()
            if stat_name == "Rush-Yds-TDs":
                # Extract rushing yards from the away and home columns
                away_rushing_yards = int(row.find_all('td')[0].text.split('-')[1])  # Away team
                home_rushing_yards = int(row.find_all('td')[1].text.split('-')[1])  # Home team
            elif stat_name == "Net Pass Yards":
                # Extract passing yards from the away and home columns
                away_passing_yards = int(row.find_all('td')[0].text)  # Away team
                home_passing_yards = int(row.find_all('td')[1].text)  # Home team
    
    away_total_yards = away_rushing_yards + away_passing_yards
    home_total_yards = home_rushing_yards + home_passing_yards


if yards_table and game_info_table:
    filename = f'data/{date}_{away_team}_vs_{home_team}_game_info.csv'
    
    # Prepare the data list
    headers = ['Date', 'Away Team', 'Home Team', 'Away Team Score', 'Home Team Score', 'Away Rushing Yards', 'Away Passing Yards', \
               'Away Total Yards', 'Home Rushing Yards', 'Home Passing Yards', ' Home Total Yards', 'Start Time', 'Temperature', 'Windspeed', 'Roof', 'Surface']
    data = [[date, away_team, home_team, away_score, home_score, away_rushing_yards, away_passing_yards, \
             away_total_yards, home_rushing_yards, home_passing_yards, home_total_yards, start_time, temperature, windspeed, roof, surface]]

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header row
        writer.writerows(data)    # Write the data rows

    # print(f"Data has been written to {filename}.")
    successful_tables += 1
else:
    print("Could not load game info.")
    

scoring_table = soup.find('table', class_='stats_table', id='scoring')
if scoring_table:
    filename = f'data/{date}_{away_team}_vs_{home_team}_scoring_table.csv'
    # Extract table headers (column names)
    headers = ['Date', 'Away Team', 'Home Team', 'Quarter', 'Time', 'Scoring Team', 'Description', 'Away Team Score', 'Home Team Score']

    # Prepare the data list
    data = []

    last_quarter = ""

    # Extract each row in the table body (tbody)
    for row in scoring_table.find('tbody').find_all('tr'):
        # Extract the quarter from the <th> element, if present
        quarter = row.find('th', {'data-stat': 'quarter'})
        if quarter:
            quarter = quarter.text.strip()
        if quarter != "":
            last_quarter = quarter
        else:    
            quarter = last_quarter
        
        # Extract the columns in each row
        columns = [td.text.strip() for td in row.find_all('td')]
        
        if len(columns) == 5:  # Ensure there are exactly 5 columns of data
            time = columns[0]
            team = columns[1]
            team = shortened_team_names[team]
            description = columns[2]
            vis_team_score = columns[3]
            home_team_score = columns[4]
            
            # Append the processed row to the data list
            data.append([date, away_team, home_team, quarter, time, team, description, vis_team_score, home_team_score])

    # Step 4: Write data to CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header row
        writer.writerows(data)    # Write the data rows
    
    # print(f"Data has been written to {filename}.")
    successful_tables += 1
else:
    print("Scoring table not found.")


player_stats = soup.find('table', id='player_offense')
if player_stats:
    rows = player_stats.find_all('tr')[2:]  # Skip the header rows

    # Create the header row for the CSV
    headers = ['Date', 'Away Team', 'Home Team', 'Player', 'Player Team', 'Passing Yards', 'Passing TD', 'Passing INT', 'Receptions', 'Receiving Yards', 'Receiving TD', 'Rushing Yards', 'Rushing TD', 'Fumbles Lost']

    player_data = []

    # Extract player stats
    for row in rows:
        columns = row.find_all('td')
        if(len(columns) == 0):
            continue
        
        # Extract the relevant data for each stat
        player_name = row.find('th').text.strip()
        player_team = columns[0].text.strip()
        player_team = team_abbreviations[player_team]

        # Passing stats
        passing_yds = columns[3].text.strip()
        passing_td = columns[4].text.strip()
        passing_int = columns[5].text.strip()

        # Receiving stats
        receptions = columns[15].text.strip()
        receiving_yds = columns[16].text.strip()
        receiving_td = columns[17].text.strip()

        # Rushing stats
        rushing_yds = columns[11].text.strip()
        rushing_td = columns[12].text.strip()

        # Fumbles Lost
        fumbles_lost = columns[20].text.strip()

        # Append the player data to the player_data list
        player_data.append([date, away_team, home_team, player_name, player_team, passing_yds, passing_td, passing_int, receptions, receiving_yds, receiving_td, rushing_yds, rushing_td, fumbles_lost])
        

    filename = f'data/{date}_{away_team}_vs_{home_team}_player_stats.csv'
    # Open a CSV file to write to
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(player_data)

    # print(f"Player stats have been written to {filename}.")
    successful_tables += 1
else:
    print("Player stats table not found.")


returns_table = soup.find('table', id='returns')
if returns_table:
    rows = returns_table.find_all('tr')[2:]  # Skip the header rows

    headers = ['Date', 'Away Team', 'Home Team', 'Player', 'Player Team', 'Kick Return TD', 'Punt Return TD']

    player_data = []

    for row in rows:
        columns = row.find_all('td')
        if(len(columns) == 0):
            continue
        
        # Extract the relevant data for each stat
        player_name = row.find('th').text.strip()
        player_team = columns[0].text.strip()
        player_team = team_abbreviations[player_team]

        kick_ret_td = columns[4].text.strip()
        punt_ret_td = columns[9].text.strip()

        # Append the player data to the player_data list
        player_data.append([date, away_team, home_team, player_name, player_team, kick_ret_td, punt_ret_td])

    filename = f'data/{date}_{away_team}_vs_{home_team}_returns.csv'
    # Open a CSV file to write to
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(player_data)

    # print(f"Player returns have been written to {filename}.")
    successful_tables += 1
else:
    print("Player returns table not found.")


kicking_table = soup.find('table', id='kicking')
if kicking_table:
    rows = kicking_table.find_all('tr')[2:]  # Skip the header rows

    headers = ['Date', 'Away Team', 'Home Team', 'Player', 'Player Team', 'Extra Points Made', 'Extra Points Missed', 'Field Goals Made', 'Field Goals Missed']

    player_data = []

    for row in rows:
        columns = row.find_all('td')
        if(len(columns) == 0):
            continue

        player_name = row.find('th').text.strip()
        player_team = columns[0].text.strip()
        player_team = team_abbreviations[player_team]

        extra_points_made = columns[1].text.strip()
        extra_points_attempted = columns[2].text.strip()
        if(extra_points_made == ""):
            extra_points_made = 0
        if(extra_points_attempted == ""):
            extra_points_attempted = 0
        extra_points_missed = int(extra_points_attempted) - int(extra_points_made)

        field_goals_made = columns[3].text.strip()
        field_goals_attempted = columns[4].text.strip()
        if(field_goals_made == ""):
            field_goals_made = 0
        if(field_goals_attempted == ""):
            field_goals_attempted = 0
        field_goals_missed = int(field_goals_attempted) - int(field_goals_made)

        player_data.append([date, away_team, home_team, player_name, player_team, extra_points_made, extra_points_missed, field_goals_made, field_goals_missed])

    filename = f'data/{date}_{away_team}_vs_{home_team}_kicking.csv'
    # Open a CSV file to write to
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(player_data)

    # print(f"Player kicking have been written to {filename}.")
    successful_tables += 1
else:
    print("Player kicking table not found.")


defense_table = soup.find('table', id='player_defense')
if defense_table:
    rows = defense_table.find_all('tr')[2:]  # Skip the header rows

    headers = []
    player_data = []
    team_data = []

    for row in rows:
        columns = row.find_all('td')
        if(len(columns) == 0):
            continue

        player_name = row.find('th').text.strip()
        player_team = columns[0].text.strip()
        player_team = team_abbreviations[player_team]

        def_int = columns[1].text.strip()
        def_int_td = columns[3].text.strip()
        
        sacks = columns[6].text.strip()

        fumbles_rec = columns[12].text.strip()
        fumbles_rec_td = columns[14].text.strip()

        def_tds = int(def_int_td) + int(fumbles_rec_td)

        player_data.append([date, away_team, home_team, player_name, player_team, def_int, sacks, fumbles_rec, def_tds])

    away_ints = 0 # 5
    away_sacks = 0 # 6
    away_fumbles = 0 # 7
    away_def_tds = 0 # 8

    home_ints = 0
    home_sacks = 0
    home_fumbles = 0
    home_def_tds = 0

    for player in player_data:
        if(player[4] == away_team):
            away_ints += int(player[5])
            away_sacks += float(player[6])
            away_fumbles += int(player[7])
            away_def_tds += int(player[8])
        else:
            home_ints += int(player[5])
            home_sacks += float(player[6])
            home_fumbles += int(player[7])
            home_def_tds += int(player[8])
    
    team_data.append([date, away_team, home_team, away_team, away_ints, away_sacks, away_fumbles, away_def_tds, home_score])
    team_data.append([date, away_team, home_team, home_team, home_ints, home_sacks, home_fumbles, home_def_tds, away_score])

    headers = ['Date', 'Away Team', 'Home Team', 'Defense Team', 'Ints', 'Sacks', 'Fumbles', 'Def TDs', 'Points Allowed']

    filename = f'data/{date}_{away_team}_vs_{home_team}_defense.csv'
    
    # Open a CSV file to write to
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(team_data)

    # print(f"Team defense stats have been written to {filename}.")
    successful_tables += 1
else:
    print("Player defense table not found.")    

print(f"{date}, {away_team}, {home_team}")
if(successful_tables == 6):
    print("Success.")
else:
    print("Missing table(s).")