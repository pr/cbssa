import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as discrete_model
import matplotlib.pyplot as plt
from scipy import stats


def logistic_reg_train(x, y, const=True, weight=None, missing="delete"):
    """Train a Logistic Regression Model

    Args:
        x: pandas DataFrame, or numpy ndarray
            independent variables, each column represents one variable
            X and Y has the same index
        y: pandas DataFrame, or numpy ndarray
            dependent variable, one column
        const: boolean, default True
            Indicator for the constant term (intercept) in the fit.
        weight: 1d numpy array,
            weight of each column
        missing: "delete","nearest","mean","median",constant number
            the method to handle the missing value

    Returns:
        A logistic regression model
    """
    data_set = pd.concat([x, y], axis=1)

    if missing is "delete":
        data_set = data_set.dropna()
    elif missing is "nearest":
        data_set = data_set.fillna(method="ffill")
        data_set = data_set.fillna(method="bfile")
    elif missing is "mean":
        values = dict(data_set.mean())
        data_set = data_set.fillna(value=values)
    elif missing is "median":
        values = dict(data_set.median())
        data_set = data_set.fillna(value=values)
    else:
        try:
            const = float(missing)
        except:
            print('Error: Type of Missing. please enter one of "delete","nearest","mean","median",constant number')
            return None
        data_set = data_set.fillna(const)
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
        mdl_coeff = pd.DataFrame(data=dict(result.params), index={"Coefficients"})
        mdl_se = pd.DataFrame(data=dict(result.bse), index={"Std error"})
        mdl_pvalue = pd.DataFrame(data=dict(result.pvalues), index={"p-value"})

    except:
        mdl_coeff = pd.DataFrame(data=result.params, index=columns_name, columns={"Coefficients"}).T
        mdl_se = pd.DataFrame(data=result.bse, index=columns_name, columns={"Std error"}).T
        mdl_pvalue = pd.DataFrame(data=result.pvalues, index=columns_name, columns={"p-value"}).T

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

    print(summary_table)
    result.SummaryTable = summary_table
    pd.set_option("display.float_format", lambda a: "%.2f" % a)

    return result


def logistic_reg_predict(model, x):
    """Make prediction based on the trained logistic regression model
    make sure input X is of the same format as training X data for the model

    Args:
        model: statsmodels.discrete.discrete_model.BinaryResultsWrapper
            a logistic regression model
        x: pandas DataFrame, or numpy ndarray
            independent variables, each column represents one variable

    Returns:
        Array of predictions
    """
    if "const" in model.SummaryTable.columns.values:
        print("adding constant")
        x = sm.add_constant(x, has_constant="add")
    print(x)
    prediction = model.predict(x)

    data = pd.DataFrame(data=x, columns=list(model.SummaryTable.columns.values))
    if "const" in model.SummaryTable.columns.values:
        data = data.drop(["const"], axis=1)
    data["prediction"] = prediction

    result = data
    return result


def linear_reg_train(x, y, const=True, weight=None, missing="delete"):
    """Train a Linear Regression Model

    Args:
        x: pandas DataFrame, or numpy ndarray
            independent variables, each column represents one variable
            X and Y has the same index
        y: pandas DataFrame, or numpy ndarray
            dependent variable, one column
        const: boolean, default True
            Indicator for the constant term (intercept) in the fit.
        weight: 1d numpy array,
            weight of each column
        missing: "delete","nearest","mean","median",constant number
            the method to handle the missing value

    Returns:
        A linear regression model
    """
    data_set = pd.concat([x, y], axis=1)

    if missing is "delete":
        data_set = data_set.dropna()
    elif missing is "nearest":
        data_set = data_set.fillna(method="ffill")
        data_set = data_set.fillna(method="bfill")
    elif missing is "mean":
        values = dict(data_set.mean())
        data_set = data_set.fillna(value=values)
    elif missing is "median":
        values = dict(data_set.median())
        data_set = data_set.fillna(value=values)
    else:
        try:
            const = float(missing)
        except:
            print('Error: Type of Missing. please enter one of "delete","nearest","mean","median",constant number')
            return None
        data_set = data_set.fillna(const)
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
    if weight is None:
        model = sm.OLS(y, x)
        result = model.fit()

    try:
        mdl_coeff = pd.DataFrame(data=dict(result.params), index={"Coefficients"})
        mdl_se = pd.DataFrame(data=dict(result.bse), index={"Std error"})
        mdl_pvalue = pd.DataFrame(data=dict(result.pvalues), index={"p-value"})

    except:
        mdl_coeff = pd.DataFrame(data=result.params, index=columns_name, columns={"Coefficients"}).T
        mdl_se = pd.DataFrame(data=result.bse, index=columns_name, columns={"Std error"}).T
        mdl_pvalue = pd.DataFrame(data=result.pvalues, index=columns_name, columns={"p-value"}).T

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

    print(summary_table)
    result.SummaryTable = summary_table

    return result


def linear_reg_predict(model, x):
    """Make prediction based on the trained linear regression model
    make sure input X is of the same format as training X data for the model

    Args:
        model: statsmodels linear regression model
        x: pandas DataFrame, or numpy ndarray
            independent variables, each column represents one variable

    Returns:
        Array of predictions
    """
    if "const" in model.SummaryTable.columns.values:
        x = sm.add_constant(x)

    prediction = model.predict(x)

    data = pd.DataFrame(data=x, columns=list(model.SummaryTable.columns.values))
    if "const" in model.SummaryTable.columns.values:
        data = data.drop(["const"], axis=1)
    data["prediction"] = prediction

    result = data
    return result


def print_binned_stats(buckets, col1, col2):
    """Print the table of binned stats

    Args:
        buckets: list of float
            a list of buckets boundaries
        col1: pandas DataFrame, or numpy ndarray
            reference column
        col2: pandas DataFrame, or numpy ndarray
            data value column

    Returns:
        table of binned stats
    """
    data_dic = {}

    idx_label = []
    count = []
    avg1 = []
    avg2 = []
    stderr2 = []
    for i in range(len(buckets) - 1):
        idx_label.append("[%s,%s)" % (buckets[i], buckets[i + 1]))
        count.append(col1[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].count())
        avg1.append(col1[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].mean())
        avg2.append(col2[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].mean())
        stderr2.append(col2[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].sem() * 2)

    idx_label[-1] = ("[%s,%s]" % (buckets[i], buckets[i + 1]))

    data_dic["Bins"] = idx_label
    data_dic["Count"] = count
    data_dic["Avg " + col1.name] = avg1
    data_dic["Avg " + col2.name] = avg2
    data_dic["Stderr " + col2.name] = stderr2

    order_list = ["Bins", "Count", "Avg " + col1.name, "Avg " + col2.name, "Stderr " + col2.name]
    summary_table = pd.DataFrame(data=data_dic)[order_list]
    print(summary_table)
    return summary_table


def graph_binned_stats(binned_stats):
    """Draw the graph

    Args:
        binned_stats: pandas DataFrame
            output summary table of function Binned_stats()
    """
    col_name = list(binned_stats.columns.values)
    fig = plt.figure(figsize=(10, 8))
    plt.errorbar(
        binned_stats[col_name[2]], binned_stats[col_name[3]], yerr=binned_stats[col_name[4]], fmt=".", capsize=5
    )
    plt.show()
    return fig


def graph_binned_stats_with_prediction(
        binned_stats,
        line_x,
        line_y,
        line_style,
        line_x2=None,
        line_y2=None,
        line_style_2=None
):
    """Draw the graph

    Args:
        binned_stats: pandas DataFrame
            output summary table of function Binned_stats()
        line_x: x input to graph for predictions
        line_y: y output of prediction
        line_style: style of line
        line_x2: second x input to graph for predictions
        line_y2: second y output of prediction
        line_style_2: second style of line
    """
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

    return fig


def bayes_normal(mean, stdev, number_of_observations, sample_mean, sample_stdev):
    """Print the table of binned stats

    Args:
        mean: float
            mean of the population
        stdev: float
            standard deviation of population
        number_of_observations: int
            number of observations
        sample_mean: float
            mean of the sample
        sample_stdev: float
            standard deviation of sample

    Returns:
        table of binned stats
    """

    post_m = (mean / stdev ** 2 + number_of_observations * sample_mean / sample_stdev ** 2) / \
             (1 / stdev ** 2 + number_of_observations / sample_stdev ** 2)
    post_sd = np.sqrt(1 / (1 / stdev ** 2 + number_of_observations / sample_stdev ** 2))

    return post_m, post_sd


def rmse(error_values=None, prediction_values=None, truth=None):
    """Calculate the RMSE of each model

    Args:
        error_values: pandas Dataframe
            matrix of errors from different model, each column represents 1 series of error
        prediction_values: pandas Dataframe
            matrix of predictions from different model, each column represents 1 series of prediction
        truth: pandas Dataframe
            array of truth, 1 column

    Returns:
        table of RMSE
    """
    if error_values is not None:
        if prediction_values is None and truth is None:
            rmse_array = np.sqrt(np.mean(error_values ** 2, axis=0))
        else:
            print("Error: only define errorValues, or only define PredictionValue and Truth")
            return None
    else:
        if prediction_values is not None and truth is not None:
            rmse_array = np.sqrt(np.mean((prediction_values.transpose() - truth) ** 2))
        else:
            print("Error: only define errorValues, or only define PredictionValue and Truth")
            return None

    return rmse_array


def model_test(error_values=None, prediction_values=None, truth=None):
    """calculate the RMSE of each model and p-value matrix and result of pairwise comparison among models

    Args:
        error_values: pandas Dataframe
            matrix of errors from different model, each column represents 1 series of error
        prediction_values: pandas Dataframe
            matrix of predictions from different model, each column represents 1 series of prediction
        truth: pandas Dataframe
            array of truth, 1 column

    Returns:
        table of RMSE and p-value matrix of each model
    """
    if error_values is not None:
        if prediction_values is None and truth is None:
            rmse_array = np.sqrt(np.mean(error_values ** 2, axis=0))
            sq_err = error_values.values ** 2
            names = list(error_values.columns.values)
        else:
            print("Error: only define errorValues, or only define PredictionValue and Truth")
            return None
    else:
        if prediction_values is not None and truth is not None:
            rmse_array = np.sqrt(np.mean((prediction_values.values - truth.values) ** 2, axis=0))
            sq_err = (prediction_values.values - truth.values) ** 2
            names = list(prediction_values.columns.values)
        else:
            print("Error: only define errorValues, or only define PredictionValue and Truth")
            return None

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


def average(number_array, row_weight):
    """calculate a weighted mean

    Args:
        number_array: Array of numbers.
        row_weight: weight of each number in average

    Returns:
        A weighted mean
    """

    return np.average(number_array, weights=row_weight)


def stddev(number_array, row_weight):
    """calculate a weighted standard deviation

    Args:
        number_array: Array of numbers.
        row_weight: weight of each number in average

    Returns:
        A weighted standard deviation
    """

    return np.sqrt(np.cov(number_array, aweights=row_weight))
