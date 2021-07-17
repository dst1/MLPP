import numpy as np
import pandas as pd


def top_n(df, n=None, frac=None):
    """
    finds the top n highest scores.
    :param df: dataframe, assumed to have inds column and target column only
    :param n: int, defaults to None, number of top choices to consider
    :param frac: float, defaults to None, fraction of top choices to consider. 
    ceiling is performed to get ints
    :return: inds of top choices
    """
    assert (n is not None) or (frac is not None)

    if n is None:
        n = int(np.ceil(df.shape[0] * frac))

    which = np.argsort(-df.iloc[:, 1])[:n]
    return (df.iloc[which, 0].tolist())


def above_q(df, q):
    """returns values above a prespecified quantile for all values. 
    q should be calculated beforehand using the helper function get_quantile or any other suitable way

    :param df: dataframe, assumed to have inds column and target column only
    :param q: np.float, quantile value to use
    """
    # which = np.where(df.iloc[:, 1] > q)
    return (df.loc[df.iloc[:, 1] > q, :].iloc[:, 0].tolist())


def get_quantile(q, h5Preds_obj, path=None, column=None):
    """
    gets the quantile value of q from the specified column in h5
    :param q: float, quantile
    :param h5Preds_obj: h5Preds object to extract data from
    :param path: path to table in h5Preds_obj
    :param column: name of column to get
    :return: np.float value of quantile q
    """

    if path is None:
        path = h5Preds_obj.default_path

    if column is None:
        column = h5Preds_obj.default_col
    assert column is not None, "Column is not specified and the object doesn't have a default"

    vals = h5Preds_obj.read_all(path=path, field=column)
    return np.quantile(vals.iloc[:, 0].values, q)
