from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import datetime
import itertools


def _parse_table(soup, limit, index):
    column_headers = [th.getText() for th in soup.findAll("thead", limit=limit)[index].findAll("th")]
    data_rows = soup.findAll("tbody", limit=limit)[index].findAll("tr")[0:]
    standings_data = [[td.getText() for td in data_rows[i].findAll(["th", "td"])] for i in range(len(data_rows))]

    return column_headers, data_rows, standings_data


def mlb(season_start_year: int) -> (pd.DataFrame, pd.DataFrame):
    url = f"https://www.baseball-reference.com/leagues/majors/{season_start_year}-schedule.shtml"
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    games = soup.find("div", {"class", "section_content"}).find_all("div")

    def append_game(d):
        date = d.find("h3").getText()
        date = datetime.date.today() if "Today" in date else datetime.datetime.strptime(date, "%A, %B %d, %Y")
        date_list = [date.strftime("%m/%d/%Y"), date.strftime("%B")]
        return list(map(lambda x: (x, date_list), d.find_all("p", {"class": "game"})))

    dates_games = list(map(append_game, games))
    dates_games = list(itertools.chain(*dates_games))

    def find_teams_and_scores(date_games):
        game_result = list(map(BeautifulSoup.getText, date_games[0].find_all("a")[:2]))
        scores_regex = re.search("\\((.*)\\).*\\((.*)\\)", date_games[0].getText(), re.S)
        game_result += [np.nan, np.nan] if scores_regex is None else list(map(int, scores_regex.groups()))
        return date_games[1] + game_result

    results = pd.DataFrame(
        map(find_teams_and_scores, dates_games),
        columns=["date", "month", "away team", "home team", "away score", "home score"]
    )

    results = results.reindex(columns=["date", "month", "home team", "away team", "home score", "away score"])
    results.replace("Arizona D'Backs", "Arizona Diamondbacks", inplace=True)
    results["home margin"] = results["home score"] - results["away score"]

    url_template = "https://www.baseball-reference.com/leagues/majors/{year}-standings.shtml"
    url = url_template.format(year=season_start_year)
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")

    league_standings = soup.find_all("div", {"id": "all_standings"})
    leagues = ["AL", "NL"]
    conferences = ["East", "Central", "West"]
    standings = []
    headers = []

    for l_standings, l in zip(league_standings, leagues):
        for conf_standings, c in zip(l_standings.find_all("div", id=re.compile("all_standings_.")), conferences):
            headers = list(map(BeautifulSoup.getText, conf_standings.find("thead").find_all("th")))
            for team in conf_standings.find("tbody").find_all("tr"):
                standing = [l, c] + list(map(BeautifulSoup.getText, team.find_all(["th", "td"])))
                standings.append(standing)

    headers = ["League", "Conference"] + headers
    standings = pd.DataFrame(standings, columns=headers)
    standings[["W", "L"]] = standings[["W", "L"]].astype("int")
    standings["Tm"] = standings["Tm"].apply(lambda x: x[2:] if x[1] == "-" else x)
    standings.set_index("Tm", inplace=True)
    standings.sort_index(inplace=True)
    return results, standings


def nfl(season_start_year: int) -> (pd.DataFrame, pd.DataFrame):
    url = f"https://www.pro-football-reference.com/years/{season_start_year}/games.htm"
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")

    column_headers = [th.getText() for th in soup.findAll("thead", limit=1)[0].findAll("th")]
    column_headers[8] = "PtsW"
    column_headers[9] = "PtsL"

    data_rows = soup.findAll("tbody", limit=1)[0].findAll("tr")[0:]

    schedule_data = [[td.getText() for td in data_rows[i].findAll(["th", "td"])] for i in range(len(data_rows))]

    schedule = pd.DataFrame(schedule_data, columns=column_headers)

    schedule = schedule.loc[((schedule.Week != "Week") &
                             (schedule.Date != "Playoffs"))]

    schedule["playoff_game"] = ((schedule.Week == "Playoffs") |
                                (schedule.Week == "WildCard") |
                                (schedule.Week == "Division") |
                                (schedule.Week == "ConfChamp") |
                                (schedule.Week == "SuperBowl"))

    schedule = schedule.reset_index(drop=True)

    schedule = schedule[["Winner/tie", "", "Loser/tie", "PtsW", "PtsL", "Week", "playoff_game"]]
    schedule.columns = ["Winner/tie", "at", "box_score", "Loser/tie", "PtsW", "PtsL", "Week", "playoff_game"]

    schedule["at"] = schedule["at"] == "@"

    schedule["away team"] = schedule["Winner/tie"] * schedule["at"] + schedule["Loser/tie"] * (1 - schedule["at"])
    schedule["home team"] = schedule["Loser/tie"] * schedule["at"] + schedule["Winner/tie"] * (1 - schedule["at"])

    schedule.loc[(schedule.PtsW == ""), "PtsW"] = -1
    schedule.loc[(schedule.PtsL == ""), "PtsL"] = -1

    schedule["PtsW"] = schedule["PtsW"].astype(int)
    schedule.loc[schedule.PtsW == -1, "PtsW"] = np.nan

    schedule["PtsL"] = schedule["PtsL"].astype(int)
    schedule.loc[schedule.PtsL == -1, "PtsL"] = np.nan

    schedule["home margin"] = ((schedule["PtsW"] - schedule["PtsL"]) *
                               (-1 * schedule["at"] + (1 - schedule["at"])))

    schedule = schedule[["Week", "playoff_game", "away team", "home team", "home margin"]]

    url = f"https://www.pro-football-reference.com/years/{season_start_year}/"
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")

    column_headers_afc, data_rows_afc, standings_data_afc, = _parse_table(soup=soup, limit=1, index=0)
    column_headers_nfc, data_rows_nfc, standings_data_nfc, = _parse_table(soup=soup, limit=2, index=1)

    standings_afc = pd.DataFrame(standings_data_afc, columns=column_headers_afc)
    standings_nfc = pd.DataFrame(
        standings_data_nfc,
        columns=column_headers_nfc,
        index=np.arange(
            len(data_rows_afc),
            len(data_rows_nfc) +
            len(data_rows_afc)
        )
    )

    standings = pd.concat([standings_afc, standings_nfc])

    divisions = standings.loc[standings["W"].isnull()][["Tm"]]
    divisions.columns = ["division"]

    standings = standings.loc[(~standings["W"].isnull())]

    standings = pd.merge(standings,
                         divisions, how="outer", left_index=True, right_index=True)

    standings["division"] = standings["division"].fillna(method="ffill")

    standings = standings.dropna()

    standings = standings.reset_index(drop=True)

    if not ("T" in standings.columns):
        standings["T"] = 0

    standings = standings[["Tm", "W", "L", "T", "division"]]
    standings["Tm"] = standings["Tm"].str.replace("[^A-Za-z0-9- ]", "")
    standings["conference"] = standings["division"].str[0:4]
    standings = standings.set_index("Tm")

    standings["W"] = standings["W"].astype(int)
    standings["L"] = standings["L"].astype(int)
    standings["T"] = standings["T"].astype(int)

    return schedule, standings


def nba(season_start_year: int) -> (pd.DataFrame, pd.DataFrame):
    year = season_start_year + 1

    schedule = pd.DataFrame(
        columns=["Date", "month", "home team", "away team", "home score", "away score", "playoff_game"])

    url_tmp = f"https://www.basketball-reference.com/leagues/NBA_{year}_games.html"

    html_tmp = urlopen(url_tmp)
    soup_tmp = BeautifulSoup(html_tmp, "html.parser")
    schedule_month = [a.getText() for a in soup_tmp.findAll("div", class_="filter")[0].findAll("a")]

    for month in schedule_month:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month.lower()}.html"

        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        data_rows = soup.findAll("tbody", limit=1)[0].findAll("tr")[0:]
        game_data = [[td.getText() for td in data_rows[i].findAll(["th", "td"])] for i in range(len(data_rows))]

        month_df = pd.DataFrame(game_data)

        if season_start_year > 2000:

            month_df = month_df.rename(
                columns={0: "Date", 2: "away team", 3: "away score", 4: "home team", 5: "home score"})

        else:
            month_df = month_df.rename(
                columns={0: "Date", 1: "away team", 2: "away score", 3: "home team", 4: "home score"})

        month_df["playoff_game"] = ((month.lower() == "june") | (month.lower() == "may"))

        if (month.lower() == "april") & (month_df.Date == "Playoffs").any():
            index_playoff = month_df.loc[month_df.Date == "Playoffs"].index
            month_df.loc[month_df.index > index_playoff[0], "playoff_game"] = 1
            month_df = month_df.loc[month_df.Date != "Playoffs"]

        month_df["playoff_game"] = month_df["playoff_game"].astype(int)

        month_df["month"] = month

        schedule = pd.concat(
            [schedule,
             month_df[[
                 "Date",
                 "month",
                 "home team",
                 "away team",
                 "home score",
                 "away score",
                 "playoff_game"
             ]]
             ],
            axis=0, ignore_index=True, sort=False
        )

    schedule.loc[((schedule["home score"] == "") | (schedule["home score"] is None)), "home score"] = -1
    schedule.loc[((schedule["away score"] == "") | (schedule["away score"] is None)), "away score"] = -1

    schedule = schedule.fillna(-1)

    schedule["home score"] = schedule["home score"].astype(int)
    schedule.loc[schedule["home score"] == -1, "home score"] = np.nan

    schedule["away score"] = schedule["away score"].astype(int)
    schedule.loc[schedule["away score"] == -1, "away score"] = np.nan

    schedule["home margin"] = schedule["home score"] - schedule["away score"]

    url_template = "https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"

    url = url_template.format(year=season_start_year)
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")

    column_headers_east, data_rows_east, standings_data_east, = _parse_table(soup=soup, limit=1, index=0)
    column_headers_west, data_rows_west, standings_data_west, = _parse_table(soup=soup, limit=2, index=1)

    standings_west = pd.DataFrame(standings_data_west, columns=column_headers_west)
    standings_west = standings_west[["Western Conference", "W", "L"]]
    standings_west.columns = ["Tm", "W", "L"]
    standings_west["conference"] = "west"

    standings_east = pd.DataFrame(standings_data_east, columns=column_headers_east)
    standings_east = standings_east[["Eastern Conference", "W", "L"]]

    standings_east.columns = ["Tm", "W", "L"]

    standings_east["conference"] = "east"

    standings = pd.concat((standings_west, standings_east), sort=False)
    standings["Tm"] = standings["Tm"].str.replace("[^A-Za-z- ]", "")

    standings.loc[standings.Tm == "Philadelphia ers", "Tm"] = "Philadelphia 76ers"
    standings = standings.set_index("Tm")
    standings = standings.dropna()
    standings["W"] = standings["W"].astype(int)

    standings["L"] = standings["L"].astype(int)

    return schedule, standings
