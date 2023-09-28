import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse import linalg


def team_train(
        home_team: pd.Series,
        away_team: pd.Series,
        home_score: pd.Series,
        away_score: pd.Series,
        row_include: bool = None
) -> (pd.DataFrame, list, float):

    if row_include is not None:
        home_team = home_team[np.array(row_include).astype(bool)]
        away_team = away_team[np.array(row_include).astype(bool)]
        home_score = home_score[np.array(row_include).astype(bool)]
        away_score = away_score[np.array(row_include).astype(bool)]

    data_size = len(home_team)

    teams = pd.Series(pd.concat([home_team, away_team], axis=0).unique())
    idx = {v: k for (k, v) in teams.to_dict().items()}
    rows = np.array([np.arange(data_size) for _ in range(3)]).flatten()

    cols = np.array(
        [np.ones(data_size) * len(teams), home_team.replace(idx).to_numpy(), away_team.replace(idx).to_numpy()]
    ).flatten()

    data = np.array([np.ones(data_size), np.ones(data_size), -np.ones(data_size)]).flatten()
    x_mat = sparse.coo_matrix((data, (rows, cols)), shape=(data_size, len(teams) + 1))

    y = home_score.to_numpy() - away_score.to_numpy()
    x, _, _, r1norm = linalg.lsqr(x_mat, y)[:4]

    home_advantage = x[-1]
    ratings = pd.DataFrame(np.array([teams.to_numpy(), x[:-1]]).T, columns=["Team", "Rating"])
    ratings.set_index("Team", inplace=True)
    ratings.sort_index(inplace=True)
    return ratings, home_advantage, r1norm / data_size ** 0.5


# Beware, not working as it seems like it should
def team_predict(
        home_team: pd.Series,
        away_team: pd.Series,
        ratings: pd.DataFrame,
        home_advantage: float = 0,
        constant: float = 0
) -> pd.DataFrame:

    df_pred = pd.concat([home_team, away_team], axis=1)
    unique_teams = np.union1d(home_team.unique(), away_team.unique())
    nonexistent_teams = np.setdiff1d(unique_teams, ratings.index.to_numpy())
    zero_df = pd.DataFrame({'Rating': np.zeros(len(nonexistent_teams))}, index=nonexistent_teams)
    ratings = pd.concat([ratings, zero_df])

    df_pred["prediction"] = constant + ratings.loc[home_team].values + home_advantage - ratings.loc[away_team].values

    return df_pred


def train(
        target: any,
        categorical_factors: pd.DataFrame,
        float_factors: pd.DataFrame = None,
        row_include: bool = None
) -> (pd.DataFrame, float):

    if row_include is not None:
        target = target[np.array(row_include).astype(bool)]
        categorical_factors = categorical_factors[np.array(row_include).astype(bool)]
        float_factors = float_factors[np.array(row_include).astype(bool)]

    data_size = len(categorical_factors.index)

    sparse_mats = []
    col_names = []
    col_type = []
    for cat_col_name in categorical_factors.columns:
        this_factor_col = categorical_factors[cat_col_name]
        this_factor_cats = pd.Series(this_factor_col.unique())
        idx = {v: k for (k, v) in this_factor_cats.to_dict().items()}
        rows = np.arange(data_size)
        cols = this_factor_col.replace(idx).to_numpy()
        data = np.ones(data_size)
        this_sparse_mat = sparse.coo_matrix((data, (rows, cols)), shape=(data_size, len(this_factor_cats)))
        sparse_mats.append(this_sparse_mat)
        col_names = col_names + this_factor_cats.tolist()
        col_type = col_type + len(this_factor_cats) * [cat_col_name]

    if float_factors is not None:
        rows = np.repeat(np.arange(data_size), len(float_factors.columns)).flatten()
        cols = np.repeat(np.arange(len(float_factors.columns)), data_size).flatten()
        data = float_factors.values.flatten()
        this_sparse_mat = sparse.coo_matrix((data, (rows, cols)), shape=(data_size, len(float_factors.columns)))
        sparse_mats.append(this_sparse_mat)
        col_names = col_names + len(float_factors.columns) * ["numeric"]
        col_type = col_type + float_factors.columns.to_list()

    rows = np.arange(data_size)
    cols = np.repeat(np.arange(1), data_size)
    data = np.ones(data_size)
    this_sparse_mat = sparse.coo_matrix((data, (rows, cols)), shape=(data_size, 1))
    sparse_mats.append(this_sparse_mat)
    col_names = col_names + ["numeric"]
    col_type = col_type + ["constant"]

    x_mat = sparse.hstack(sparse_mats)
    y = target.to_numpy()
    x, _, _, r1norm = linalg.lsqr(x_mat, y)[:4]

    model_coefficients = pd.DataFrame({"type": col_type, "name": col_names, "coeff": x})
    for col in categorical_factors.columns:
        mean = model_coefficients.loc[model_coefficients['type'] == col]['coeff'].mean()
        model_coefficients.loc[model_coefficients['type'] == 'constant', 'coeff'] += mean
        model_coefficients.loc[model_coefficients['type'] == col, 'coeff'] -= mean

    return model_coefficients, r1norm / data_size ** 0.5


def predict(
        model_coefficients: pd.DataFrame,
        df: pd.DataFrame
) -> pd.DataFrame:
    unique_types = model_coefficients.loc[:, "type"].unique().tolist()

    df_pred = df.copy(deep=True)

    df_pred["constant"] = 1
    df_pred = df_pred.loc[:, unique_types]

    for unique_type in unique_types:
        tmp_df = model_coefficients.loc[model_coefficients["type"] == unique_type, ["name", "coeff"]]

        if tmp_df.iloc[0]["name"] == "numeric":
            df_pred[unique_type] = df_pred[unique_type] * tmp_df["coeff"].values
        else:
            this_col_new_data = dict(zip(tmp_df["name"], tmp_df["coeff"]))
            df_pred[unique_type] = df_pred[unique_type].replace(this_col_new_data).to_numpy()

    df_pred = df_pred.apply(pd.to_numeric, axis=0, errors="coerce")
    df_pred = df_pred.replace(np.nan, 0)

    df["prediction"] = df_pred.sum(axis=1)

    return df


def to_probabilities(
        ratings: pd.Series,
        rmse: float,
        schedule: pd.DataFrame,
        home_advantage: float
) -> pd.DataFrame:
    schedule_temp = schedule.copy()

    schedule_temp = pd.merge(schedule_temp, ratings, left_on=["home team"], right_index=True)
    schedule_temp = pd.merge(
        schedule_temp, ratings, left_on=["away team"], right_index=True, suffixes=("_home", "_away")
    )

    schedule_temp['pred margin'] = schedule_temp['Rating_home'] - schedule_temp['Rating_away'] + home_advantage
    schedule_temp['prob_home'] = 1 - scipy.stats.norm.cdf(0, schedule_temp['pred margin'], rmse)
    schedule_temp.sort_index(inplace=True)

    return schedule_temp
