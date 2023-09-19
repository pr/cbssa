import pandas as pd
import numpy as np

import cbssa


def get_data():
    return pd.read_csv("test_data/logitRegData.csv")


def get_model():
    data = get_data()

    logit_reg_x = np.stack([data.dist.values, data.dist.values ** 2 / 100, data.dist.values ** 3 / 1000]).T
    data_logit_reg_x = pd.DataFrame(data=logit_reg_x, columns=['dist', 'dist^2/100', 'dist^3/1000'])
    data_logit_reg_y = data.make

    mdl = cbssa.logistic_regression.train(data_logit_reg_x, data_logit_reg_y)
    mdl_legacy = cbssa.legacy.LogisticRegTrain(data_logit_reg_x, data_logit_reg_y)

    return mdl, mdl_legacy


def get_bins_set():
    bins_set = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    include_se = True
    se_multiplier_set = 2

    return bins_set, include_se, se_multiplier_set


class TestBackwardsCompatibility:

    def test_logistic_reg_train(self):
        mdl, mdl_legacy = get_model()

        assert mdl.summary_table.at["Coefficients", "const"] == 14.043798183157614
        assert mdl.summary_table.at["Coefficients", "dist^3/1000"] == -0.12182629636884296

        assert mdl_legacy.SummaryTable.at["Coefficients", "const"] == 14.043798183157614
        assert mdl_legacy.SummaryTable.at["Coefficients", "dist^3/1000"] == -0.12182629636884296

    def test_logistic_reg_predict(self):
        mdl, mdl_legacy = get_model()

        pred_x = pd.read_excel("test_data/predictionData.xlsx", sheet_name="predData")

        prediction = cbssa.logistic_regression.predict(mdl, pred_x)
        prediction_legacy = cbssa.legacy.LogisticRegPredict(mdl_legacy, pred_x)

        assert prediction.at[0, "prediction"] == 0.9973873291026644
        assert prediction.at[2, "dist^2/100"] == 2.89
        assert prediction.at[8, "prediction"] == 0.9747404284857288

        assert prediction_legacy.at[0, "prediction"] == 0.9973873291026644
        assert prediction_legacy.at[2, "dist^2/100"] == 2.89
        assert prediction_legacy.at[8, "prediction"] == 0.9747404284857288

    def test_get_binned_stats(self):
        data = get_data()
        bins_set, include_se, se_multiplier_set = get_bins_set()

        summary_table = cbssa.data_binning.get(bins_set, data.dist, data.make)
        summary_table_legacy = cbssa.legacy.Binned_stats(bins_set, data.dist, data.make, include_se, se_multiplier_set)

        assert summary_table.at[2, "Bins"] == "[25,30)"
        assert summary_table.at[8, "Avg make"] == 0.4492753623188406

        assert summary_table_legacy.at[2, "Bins"] == "[25,30)"
        assert summary_table_legacy.at[8, "Avg make"] == 0.4492753623188406
