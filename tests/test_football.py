import pandas as pd
import numpy as np

from src import cbssa
from src.cbssa.football import Down
from src.cbssa.football import FIELD_OF_PLAY_LENGTH


class TestFootball:

    def test_expected_points(self):
        expected_points = cbssa.football.expected_points(
            down=Down.FIRST,
            yards_to_go=5,
            yards_from_own_goal=45
        )
        assert expected_points == 2.6869588906927384

        another_expected_points = cbssa.football.expected_points(
            down=Down.SECOND,
            yards_to_go=2,
            yards_from_own_goal=48
        )
        assert another_expected_points == 2.608165103462263

    def test_expected_points_going_for_it_fourth_down(self):
        result = cbssa.football.expected_points_going_for_it_fourth_down(61, 5)
        assert result == 0.5927685647611183

    def test_go_for_it_results(self):
        results_df = pd.DataFrame(columns=np.arange(20, FIELD_OF_PLAY_LENGTH, 5), index=np.arange(1, 13))

        for yards_from_own_goal in np.arange(20, FIELD_OF_PLAY_LENGTH, 5):
            for yards_to_go in np.arange(1, 13):
                expected_points_series = pd.Series(index=["Go", "FG", "Punt"])

                if yards_from_own_goal + yards_to_go <= FIELD_OF_PLAY_LENGTH:
                    expected_points_series["Go"] = cbssa.football.expected_points_going_for_it_fourth_down(
                        yards_from_own_goal, yards_to_go
                    )
                    expected_points_series["FG"] = cbssa.football.expected_points_field_goal(yards_from_own_goal)
                    expected_points_series["Punt"] = cbssa.football.expected_points_punt(yards_from_own_goal)

                    results_df.loc[yards_to_go, yards_from_own_goal] = expected_points_series.idxmax()

        assert results_df[30][8] == "Punt"
        assert results_df[90][5] == "FG"
        assert results_df[95][1] == "Go"
