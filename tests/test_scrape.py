from src import cbssa


class TestScrape:

    def test_mlb(self):
        schedule, standings = cbssa.scrape.mlb(2000)

        assert standings.at["Arizona Diamondbacks", "W"] == 85

    def test_nfl(self):
        schedule, standings = cbssa.scrape.nfl(2000)

        assert standings.at["New York Jets", "W"] == 9

    def test_nba(self):
        schedule_1996, standings_1996 = cbssa.scrape.nba(1996)
        assert standings_1996.at["Chicago Bulls*", "W"] == 72

        schedule_2020, standings_2020 = cbssa.scrape.nba(2020)
        assert standings_2020.at["New York Knicks", "W"] == 21
