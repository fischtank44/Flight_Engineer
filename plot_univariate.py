import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

#from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline)

from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)

from regression_tools.plotting_tools import (
    plot_univariate_smooth,
    bootstrap_train,
    display_coef,
    plot_bootstrap_coefs,
    plot_partial_depenence,
    plot_partial_dependences,
    predicteds_vs_actuals)



def plot_one_univariate(ax, df, y, var_name, mask=None, bootstrap=100):
    if mask is None:
        plot_univariate_smooth(
            ax,
            df[var_name].values.reshape(-1, 1), 
            df[y],
            bootstrap=bootstrap)
    else:
        plot_univariate_smooth(
            ax,
            df[var_name].values.reshape(-1, 1), 
            df[y],
            mask=mask,
            bootstrap=bootstrap)
