import pandas as pd
import numpy as np

from src.sports_analytics import sports_analytics
from src.sports_analytics import legacy


class TestSportsAnalytics:
    def test_logistic_reg_train(self):
        data = pd.read_csv("test_data.csv")
        data.head()

        logit_reg_x = np.stack([data.dist.values, data.dist.values ** 2 / 100, data.dist.values ** 3 / 1000]).T
        data_logit_reg_x = pd.DataFrame(data=logit_reg_x, columns=['dist', 'dist^2/100', 'dist^3/1000'])
        data_logit_reg_y = data.make

        mdl = sports_analytics.logistic_reg_train(data_logit_reg_x, data_logit_reg_y)
        mdl_legacy = legacy.LogisticRegTrain(data_logit_reg_x, data_logit_reg_y)

        assert mdl.SummaryTable.at["Coefficients", "const"] == 16.429114791522277
        assert mdl.SummaryTable.at["Coefficients", "dist^3/1000"] == -0.14254162236818144
        assert mdl_legacy.SummaryTable.at["Coefficients", "const"] == 16.429114791522277
        assert mdl_legacy.SummaryTable.at["Coefficients", "dist^3/1000"] == -0.14254162236818144
