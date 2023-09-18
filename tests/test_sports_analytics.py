import pandas as pd
import numpy as np

from src.cbssa import sports_analytics
from src.cbssa import legacy


def get_model():
    data = pd.read_csv("test_data/logitRegData.csv")

    logit_reg_x = np.stack([data.dist.values, data.dist.values ** 2 / 100, data.dist.values ** 3 / 1000]).T
    data_logit_reg_x = pd.DataFrame(data=logit_reg_x, columns=['dist', 'dist^2/100', 'dist^3/1000'])
    data_logit_reg_y = data.make

    mdl = sports_analytics.logistic_reg_train(data_logit_reg_x, data_logit_reg_y)
    mdl_legacy = legacy.LogisticRegTrain(data_logit_reg_x, data_logit_reg_y)

    return mdl, mdl_legacy


class TestSportsAnalytics:

    def test_logistic_reg_train(self):
        mdl, mdl_legacy = get_model()

        assert mdl.SummaryTable.at["Coefficients", "const"] == 14.043798183157614
        assert mdl.SummaryTable.at["Coefficients", "dist^3/1000"] == -0.12182629636884296

        assert mdl_legacy.SummaryTable.at["Coefficients", "const"] == 14.043798183157614
        assert mdl_legacy.SummaryTable.at["Coefficients", "dist^3/1000"] == -0.12182629636884296

    def test_logistic_reg_predict(self):
        mdl, mdl_legacy = get_model()

        pred_x = pd.read_excel("test_data/predictionData.xlsx", sheet_name="predData")

        prediction = sports_analytics.logistic_reg_predict(mdl, pred_x)
        prediction_legacy = legacy.LogisticRegPredict(mdl_legacy, pred_x)

        assert prediction.at[0, "prediction"] == 0.9973873291026644
        assert prediction.at[2, "dist^2/100"] == 2.89
        assert prediction.at[8, "prediction"] == 0.9747404284857288

        assert prediction_legacy.at[0, "prediction"] == 0.9973873291026644
        assert prediction_legacy.at[2, "dist^2/100"] == 2.89
        assert prediction_legacy.at[8, "prediction"] == 0.9747404284857288
