import numpy as np
import pandas as pd
from scipy import stats


def rmse(
        error_values: pd.DataFrame = None,
        prediction_values: pd.DataFrame = None,
        truth: pd.DataFrame = None
) -> np.ndarray:
    if error_values is not None:
        if prediction_values is None and truth is None:
            rmse_array = np.sqrt(np.mean(error_values ** 2, axis=0))
        else:
            raise Exception("Only define errorValues, or only define PredictionValue and Truth")
    else:
        if prediction_values is not None and truth is not None:
            rmse_array = np.sqrt(np.mean((prediction_values.transpose() - truth) ** 2))
        else:
            raise Exception("Only define errorValues, or only define PredictionValue and Truth")

    return rmse_array


def model_test(
        error_values: pd.DataFrame = None,
        prediction_values: pd.DataFrame = None,
        truth: pd.DataFrame = None,
        print_table: bool = False
) -> pd.DataFrame:
    if error_values is not None:
        if prediction_values is None and truth is None:
            rmse_array = np.sqrt(np.mean(error_values ** 2, axis=0))
            sq_err = error_values.values ** 2
            names = list(error_values.columns.values)
        else:
            raise Exception("Only define errorValues, or only define PredictionValue and Truth")
    else:
        if prediction_values is not None and truth is not None:
            rmse_array = np.sqrt(np.mean((prediction_values.values - truth.values) ** 2, axis=0))
            sq_err = (prediction_values.values - truth.values) ** 2
            names = list(prediction_values.columns.values)
        else:
            raise Exception("Only define errorValues, or only define PredictionValue and Truth")

    pvalue_matrix = np.empty(shape=(sq_err.shape[1], sq_err.shape[1]))
    pvalue_matrix[:] = np.nan

    for eachCol in range(sq_err.shape[1]):
        for eachCol2 in range(eachCol + 1, sq_err.shape[1]):
            tmp_t, tmp_p = stats.ttest_rel(sq_err[:, eachCol], sq_err[:, eachCol2])
            pvalue_matrix[eachCol, eachCol2] = 1 - tmp_p / 2
            pvalue_matrix[eachCol2, eachCol] = tmp_p / 2

    summary_table = pd.DataFrame(data=pd.DataFrame(np.concatenate([rmse_array[:, None], pvalue_matrix], axis=1).T))
    summary_table.columns = names
    summary_table.index = ["RMSE"] + names

    if print_table:
        print(summary_table)

    summary_table = summary_table.fillna("")
    return summary_table
