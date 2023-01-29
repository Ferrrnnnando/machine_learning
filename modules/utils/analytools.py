import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


### Data Cleaning ###
def get_null_cols(data):
    """Get columns that contains null value"""
    return data.columns[data.count() != len(data)]


def handle_outliers(
    data,
    cols=[],
    method="capping",
    lower_quantile=0.25,
    upper_quantile=0.75,
    whisker_width=1.5,
):
    """Handle outliers

    Parameters
    ----------
    data : dataFrame object
    cols: list of str
        List of column names of `data` to be handled outliers.
    method: str
        Two methods are provided: `capping` and `drop`. `capping` will constrain outliers
        to lower/upper whisker values. `drop` will drop the entire row from `data`.

    Returns
    -------
    data_clean: dataFrame object
        A copy of `data`. Clean data with outliers handled.
    """
    data_clean = data.copy()

    for col in cols:
        q1 = data_clean[col].quantile(lower_quantile)
        q3 = data_clean[col].quantile(upper_quantile)
        iqr = q3 - q1

        lower_whisker = q1 - whisker_width * iqr
        upper_whisker = q3 + whisker_width * iqr

        if method == "cap":
            data_clean[col] = np.where(
                data[col] > upper_whisker,
                upper_whisker,
                np.where(data[col] < lower_whisker, lower_whisker, data[col]),
            )
        elif method == "drop":
            data_clean = data_clean.drop(
                data_clean[
                    (data_clean[col] < lower_whisker)
                    | (data_clean[col] > upper_whisker)
                ].index
            )

    return data_clean


### Data Analytics ###
def make_mi_scores(X, y, discrete_features):
    # discrete_features: boolean mask that indicate which features are discrete, then the ramaining is continuous
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores  # same shape as X.shape[0]


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
