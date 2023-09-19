import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.regression.linear_model as linear_model

from cbssa.utils import MissingValueAction


def train(
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

    pd.set_option("display.float_format", lambda a: "%.4f" % a)
    summary_table = summary_table.fillna("")

    try:
        summary_table.index.name = y.name
    except:
        pass

    if print_table:
        print(summary_table)

    result.summary_table = summary_table

    return result


def predict(
        model: linear_model.RegressionResultsWrapper,
        x: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:

    if "const" in model.summary_table.columns.values:
        x = sm.add_constant(x)

    prediction = model.predict(x)

    result = pd.DataFrame(data=x, columns=list(model.summary_table.columns.values))
    if "const" in model.summary_table.columns.values:
        result = result.drop(["const"], axis=1)

    result["prediction"] = prediction

    return result
