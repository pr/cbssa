import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as discrete_model
import statsmodels.regression.linear_model as linear_model
import matplotlib.figure
import matplotlib.pyplot as plt
from enum import Enum
from scipy import stats


class MissingValueAction(Enum):
    DELETE = "delete"
    NEAREST = "nearest"
    MEAN = "mean"
    MEDIAN = "median"
    FILL_VALUE = "fill_value"


def logistic_reg_train(
        x: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        const: bool = True,
        weight: np.array = None,
        missing: MissingValueAction = MissingValueAction.DELETE,
        missing_fill_value: float = None,
        print_table: bool = False,
) -> discrete_model.BinaryResultsWrapper:

    data_set = pd.concat([x, y], axis=1)

    match missing:
        case MissingValueAction.DELETE:
            data_set = data_set.dropna()
        case MissingValueAction.NEAREST:
            data_set = data_set.fillna(method="ffill")
            data_set = data_set.fillna(method="bfile")
        case MissingValueAction.MEAN:
            values = dict(data_set.mean())
            data_set = data_set.fillna(value=values)
        case MissingValueAction.MEDIAN:
            values = dict(data_set.median())
            data_set = data_set.fillna(value=values)
        case MissingValueAction.FILL_VALUE:
            if missing_fill_value is None or type(missing_fill_value) != float:
                raise Exception("if 'missing' is 'fill_value', then pass a float 'missing_fill_value'")
            data_set = data_set.fillna(missing_fill_value)
        case _:
            raise Exception("parameter 'missing' has invalid value")

    x = data_set[data_set.columns.values[:-1]]
    y = data_set[data_set.columns.values[-1]]

    if weight is not None:
        x = pd.DataFrame(data=x.values * weight, columns=x.columns.values, index=x.index)

    if const is True:
        x = sm.add_constant(x)
        columns_name = ["const"] + ["x%s" % n for n in range(1, x.shape[1])]
    else:
        columns_name = ["x%s" % n for n in range(1, x.shape[1])]

    model = discrete_model.Logit(y, x)
    result = model.fit()

    try:
        mdl_coeff = pd.DataFrame(data=dict(result.params), index=["Coefficients"])
        mdl_se = pd.DataFrame(data=dict(result.bse), index=["Std error"])
        mdl_pvalue = pd.DataFrame(data=dict(result.pvalues), index=["p-value"])
    except:
        mdl_coeff = pd.DataFrame(data=result.params, index=columns_name, columns=["Coefficients"]).T
        mdl_se = pd.DataFrame(data=result.bse, index=columns_name, columns=["Std error"]).T
        mdl_pvalue = pd.DataFrame(data=result.pvalues, index=columns_name, columns=["p-value"]).T

    summary_table = pd.concat((mdl_coeff, mdl_se, mdl_pvalue))
    summary_table.loc["Log-likelihood", summary_table.columns.values[0]] = result.llf
    summary_table.loc["Number valid obs", summary_table.columns.values[0]] = result.df_resid
    summary_table.loc["Total obs", summary_table.columns.values[0]] = result.nobs

    pd.set_option("display.float_format", lambda a: "%.4f" % a)
    summary_table = summary_table.fillna("")

    try:
        summary_table.index.name = y.name
    except:
        pass

    if print_table:
        print(summary_table)

    result.SummaryTable = summary_table
    pd.set_option("display.float_format", lambda a: "%.2f" % a)

    return result


def logistic_reg_predict(
        model: discrete_model.BinaryResultsWrapper,
        x: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:

    if "const" in model.SummaryTable.columns.values:
        x = sm.add_constant(x, has_constant="add")

    prediction = model.predict(x)

    result = pd.DataFrame(data=x, columns=list(model.SummaryTable.columns.values))
    if "const" in model.SummaryTable.columns.values:
        result = result.drop(["const"], axis=1)
    result["prediction"] = prediction

    return result


def linear_reg_train(
        x: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        const: bool = True,
        weight: np.array = None,
        missing: MissingValueAction = MissingValueAction.DELETE,
        missing_fill_value: float = None,
        print_table=False,
) -> linear_model.RegressionResults:

    data_set = pd.concat([x, y], axis=1)

    match missing:
        case MissingValueAction.DELETE:
            data_set = data_set.dropna()
        case MissingValueAction.NEAREST:
            data_set = data_set.fillna(method="ffill")
            data_set = data_set.fillna(method="bfile")
        case MissingValueAction.MEAN:
            values = dict(data_set.mean())
            data_set = data_set.fillna(value=values)
        case MissingValueAction.MEDIAN:
            values = dict(data_set.median())
            data_set = data_set.fillna(value=values)
        case MissingValueAction.FILL_VALUE:
            if missing_fill_value is None or type(missing_fill_value) != float:
                raise Exception("if 'missing' is 'fill_value', then pass a float 'missing_fill_value'")
            data_set = data_set.fillna(missing_fill_value)
        case _:
            raise Exception("parameter 'missing' has invalid value")

    x = data_set[data_set.columns.values[:-1]]
    y = data_set[data_set.columns.values[-1]]

    if const is True:
        x = sm.add_constant(x)
        columns_name = ["const"] + ["x%s" % n for n in range(1, x.shape[1])]
    else:
        columns_name = ["x%s" % n for n in range(1, x.shape[1])]

    if weight is not None:
        model = sm.WLS(y, x, weights=weight)
        result = model.fit()
    else:
        model = sm.OLS(y, x)
        result = model.fit()

    try:
        mdl_coeff = pd.DataFrame(data=dict(result.params), index=["Coefficients"])
        mdl_se = pd.DataFrame(data=dict(result.bse), index=["Std error"])
        mdl_pvalue = pd.DataFrame(data=dict(result.pvalues), index=["p-value"])

    except:
        mdl_coeff = pd.DataFrame(data=result.params, index=columns_name, columns=["Coefficients"]).T
        mdl_se = pd.DataFrame(data=result.bse, index=columns_name, columns=["Std error"]).T
        mdl_pvalue = pd.DataFrame(data=result.pvalues, index=columns_name, columns=["p-value"]).T

    summary_table = pd.concat((mdl_coeff, mdl_se, mdl_pvalue))
    summary_table.loc["Log-likelihood", summary_table.columns.values[0]] = result.llf
    summary_table.loc["Number valid obs", summary_table.columns.values[0]] = result.df_resid
    summary_table.loc["Total obs", summary_table.columns.values[0]] = result.nobs

    pd.set_option("display.float_format", lambda a: "%.2f" % a)
    summary_table = summary_table.fillna("")

    try:
        summary_table.index.name = y.name
    except:
        pass

    if print_table:
        print(summary_table)

    result.SummaryTable = summary_table

    return result


def linear_reg_predict(
        model: linear_model.RegressionResultsWrapper,
        x: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:

    if "const" in model.SummaryTable.columns.values:
        x = sm.add_constant(x)

    prediction = model.predict(x)

    result = pd.DataFrame(data=x, columns=list(model.SummaryTable.columns.values))
    if "const" in model.SummaryTable.columns.values:
        result = result.drop(["const"], axis=1)
    result["prediction"] = prediction

    return result


def get_binned_stats(
        buckets: list[float],
        col1: pd.DataFrame | np.ndarray,
        col2: pd.DataFrame | np.ndarray,
        print_table: bool = False
) -> pd.DataFrame:
    data_dic = {}

    idx_label = []
    count = []
    avg1 = []
    avg2 = []
    stderr2 = []
    i = None

    for i in range(len(buckets) - 1):
        idx_label.append("[%s,%s)" % (buckets[i], buckets[i + 1]))
        count.append(col1[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].count())
        avg1.append(col1[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].mean())
        avg2.append(col2[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].mean())
        stderr2.append(col2[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].sem() * 2)

    if i:
        idx_label[-1] = f"[{buckets[i]}, {buckets[i + 1]}]"

    data_dic["Bins"] = idx_label
    data_dic["Count"] = count
    data_dic[f"Avg {col1.name}"] = avg1
    data_dic[f"Avg {col2.name}"] = avg2
    data_dic[f"Stderr {col2.name}"] = stderr2

    order_list = ["Bins", "Count", f"Avg {col1.name}", f"Avg {col2.name}", f"Stderr {col2.name}"]
    summary_table = pd.DataFrame(data=data_dic)[order_list]

    if print_table:
        print(summary_table)

    return summary_table


def graph_binned_stats(
        binned_stats: pd.DataFrame,
        show_graph: bool = False,
) -> matplotlib.figure.Figure:
    """Draw the graph

    Args:
        binned_stats: pandas DataFrame
            output summary table of function Binned_stats()
        show_graph: boolean, default True
            show the graph or not
    """
    col_name = list(binned_stats.columns.values)
    fig = plt.figure(figsize=(10, 8))
    plt.errorbar(
        binned_stats[col_name[2]], binned_stats[col_name[3]], yerr=binned_stats[col_name[4]], fmt=".", capsize=5
    )
    if show_graph:
        plt.show()

    return fig


def graph_binned_stats_with_prediction(
        binned_stats: pd.DataFrame,
        line_x: pd.Series,
        line_y: pd.Series,
        line_style: str,
        line_x2: pd.Series = None,
        line_y2: pd.Series = None,
        line_style_2: str = None,
        show_graph: bool = False
) -> matplotlib.figure.Figure:

    col_name = list(binned_stats.columns.values)
    fig = plt.figure(figsize=(10, 8))
    plt.errorbar(
        binned_stats[col_name[2]], binned_stats[col_name[3]], yerr=binned_stats[col_name[4]], fmt=".", capsize=5
    )

    if line_x2 is not None:
        plt.plot(line_x, line_y, line_style, line_x2, line_y2, line_style_2)

    else:
        plt.plot(line_x, line_y, line_style)

    plt.xlabel("distance")
    plt.ylabel("make")

    if show_graph:
        plt.show()

    return fig


def bayes_normal(
        mean: float,
        standard_deviation: float,
        number_of_observations: int,
        sample_mean: float,
        sample_stdev: float,
) -> (float, float):

    post_m = (mean / standard_deviation ** 2 + number_of_observations * sample_mean / sample_stdev ** 2) / \
             (1 / standard_deviation ** 2 + number_of_observations / sample_stdev ** 2)
    post_sd = np.sqrt(1 / (1 / standard_deviation ** 2 + number_of_observations / sample_stdev ** 2))

    return post_m, post_sd


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
        truth: pd.DataFrame = None
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
    print(summary_table)
    summary_table = summary_table.fillna("")
    return summary_table


def average(
        number_array: list,
        row_weight: list,
) -> float:

    return np.average(number_array, weights=row_weight)


def stddev(
        number_array: list,
        row_weight: list,
) -> float:

    return np.sqrt(np.cov(number_array, aweights=row_weight))
