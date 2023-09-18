import pandas as pd
import pytest
import numpy as np

from src.cbssa import sports_analytics


def get_data():
    data = pd.read_csv("test_data/logitRegData.csv")
    logit_reg_x = np.stack([data.dist.values, data.dist.values ** 2 / 100, data.dist.values ** 3 / 1000]).T
    data_logit_reg_x = pd.DataFrame(data=logit_reg_x, columns=['dist', 'dist^2/100', 'dist^3/1000'])
    data_logit_reg_y = data.make

    return data, logit_reg_x, data_logit_reg_x, data_logit_reg_y


def get_bins_set():
    return [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]


class TestSportsAnalytics:

    def test_logistic_reg_train(self):
        data, logit_reg_x, data_logit_reg_x, data_logit_reg_y = get_data()
        model = sports_analytics.logistic_reg_train(data_logit_reg_x, data_logit_reg_y)

        assert model.SummaryTable.at["Coefficients", "const"] == 14.043798183157614
        assert model.SummaryTable.at["Coefficients", "dist^3/1000"] == -0.12182629636884296

    def test_logistic_reg_train_bad_missing_input(self):
        data, logit_reg_x, data_logit_reg_x, data_logit_reg_y = get_data()

        with pytest.raises(Exception) as exception:
            sports_analytics.logistic_reg_train(
                data_logit_reg_x,
                data_logit_reg_y,
                missing=sports_analytics.MissingValueAction.FILL_VALUE,
                missing_fill_value=None
            )

        assert exception.value.args[0] == "if 'missing' is 'fill_value', then pass a float 'missing_fill_value'"

    def test_logistic_reg_predict(self):
        data, logit_reg_x, data_logit_reg_x, data_logit_reg_y = get_data()
        model = sports_analytics.logistic_reg_train(data_logit_reg_x, data_logit_reg_y)

        pred_x = pd.read_excel("test_data/predictionData.xlsx", sheet_name="predData")

        prediction = sports_analytics.logistic_reg_predict(model, pred_x)

        assert prediction.at[0, "prediction"] == 0.9973873291026644
        assert prediction.at[2, "dist^2/100"] == 2.89
        assert prediction.at[8, "prediction"] == 0.9747404284857288

    def test_get_binned_stats(self):
        data, _, _, _ = get_data()

        summary_table = sports_analytics.get_binned_stats(get_bins_set(), data.dist, data.make)

        assert summary_table.at[2, "Bins"] == "[25,30)"
        assert summary_table.at[8, "Avg make"] == 0.4492753623188406

    def test_get_binned_stats_empty_list(self):
        data, _, _, _ = get_data()
        bins_set = []

        summary_table = sports_analytics.get_binned_stats(bins_set, data.dist, data.make)
        assert summary_table.empty

    def test_graph_binned_stats(self):
        data, _, _, _ = get_data()

        summary_table = sports_analytics.get_binned_stats(get_bins_set(), data.dist, data.make)

        sports_analytics.graph_binned_stats(summary_table)

    def test_graph_binned_stats_with_prediction(self):
        data, logit_reg_x, data_logit_reg_x, data_logit_reg_y = get_data()
        model = sports_analytics.logistic_reg_train(data_logit_reg_x, data_logit_reg_y)

        pred_x = pd.read_excel("test_data/predictionData.xlsx", sheet_name="predData")

        summary_table = sports_analytics.get_binned_stats(get_bins_set(), data.dist, data.make)
        prediction = sports_analytics.logistic_reg_predict(model, pred_x)

        sports_analytics.graph_binned_stats_with_prediction(summary_table, prediction.dist, prediction.prediction, "")

    def test_bayes_normal(self):
        mean, standard_deviation = sports_analytics.bayes_normal(0.3, 4, 5, 2.8, 1)
        assert mean == 2.769135802469136
        assert standard_deviation == 0.4444444444444444

    def test_average(self):
        number_array = [2, 4]
        row_weight = [2, 8]

        result = sports_analytics.average(number_array, row_weight)

        assert result == 3.6

    def test_stddev(self):
        number_array = [2, 4]
        row_weight = [2, 8]

        result = sports_analytics.stddev(number_array, row_weight)

        assert result == 1.4142135623730951
