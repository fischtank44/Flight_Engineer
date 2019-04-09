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
import numpy.random





x = np.arange(-5, 5, .1)
x = list(x)
y= []
for num in x:
    y.append(1 + num + max( 0, num-1) + np.random.normal(loc=0, scale=.5) )

y
len(y)
len(x)

df = pd.DataFrame(columns= ['y', 'x'])
df['x'] = x
df['y'] = y


df

x_fit = Pipeline([
    ('x', ColumnSelector(name='x')),
    ('x_spline', LinearSpline(knots=[-5, 1]))
])



feature_pipeline = FeatureUnion([
    ('x', x_fit)
])




feature_pipeline.fit(df['x'])
features = feature_pipeline.transform(df)
features

model = LinearRegression(fit_intercept=True)
model.fit(features, df['y'] ) 

model.coef_



type(features)
features.columns[0]
features.columns[1]
features.columns[2]
features.columns[3]




for _ in range(len(model.coef_)):
    print(str(features.columns[_]) + " coef " + 
    str(model.coef_[_]))


for _ in x_fit:
    print(_)


feat = [i for i in features.columns] 
coef = [j for j in float(model.coef_)]
print(zip( feat , coef  ) )




# printing players and scores. 
for fe, co in zip(feat, coef): 
    print(fe, co)
    print ("Feat :  %s     Coef : %f" %(fe, co)) 




