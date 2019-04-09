python

import numpy as np
import pandas as pd
# from pandas.plotting import scatter_matrix

from regression_tools.dftransformers import (
    ColumnSelector, 
    Identity,
    Intercept,
    FeatureUnion, 
    MapFeature,
    StandardScaler)
from scipy import stats
from plot_univariate import plot_one_univariate
from sklearn.linear_model import (
    LinearRegression,
)
from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline)
from regression_tools.plotting_tools import (
    plot_univariate_smooth,
    bootstrap_train,
    display_coef,
    plot_bootstrap_coefs,
    plot_partial_depenence,
    plot_partial_dependences,
    predicteds_vs_actuals)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import log_loss, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from basis_expansions.basis_expansions import NaturalCubicSpline
import random
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


x = np.arange(-5, 5, .1)
x = list(x)
y= []
for num in x:
    y.append(1 + num + max( 0, num-1))

y
len(y)
len(x)

df = pd.DataFrame(columns= ['y', 'x'])
df['x'] = x
df['y'] = y




cycle_fit = Pipeline([
    ('x', ColumnSelector(name='x')),
    ('x_spline', LinearSpline(knots=[25, 50, 75, 120, 175 , 220, 240, 260, 280, 300]))
])



feature_pipeline = FeatureUnion([
    ('time_cycles', cycle_fit)
])


