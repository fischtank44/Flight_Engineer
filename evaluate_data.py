# python
import numpy as np
import pandas as pd
# from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from basis_expansions.basis_expansions import NaturalCubicSpline
from regression_tools.dftransformers import (
    ColumnSelector, Identity,
    FeatureUnion, MapFeature,
    StandardScaler)

from regression_tools.plotting_tools import (
    plot_partial_depenence,
    plot_partial_dependences,
    predicteds_vs_actuals)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
os.getcwd()


np.random.seed(137)


############  Start of data analysis   #############


## this will plot all columns to check for any variation in the data
for name in col:
    df1.plot.scatter( 'cycles_to_fail', name, alpha = .3)
    plt.show()


#    view the description of each column 
col = df1.columns
# col = train_features
for c in col:
  print (df1[c].describe() ) 


### This will print only the standard deviation for each column
col = df1.columns
for c in col:
  print (df1[c].describe()[2] ) 


### This will remove features based the standard deviation for each column
train_features = []
limit = .01
col = df1.columns
for c in col:
  if (df1[c].describe()[2] ) >= .01:
      train_features.append(c)
train_features

#### List of features to train the model to  #######
train_features = ['unit', 't24_lpc', 't30_hpc', 't50_lpt', 
    'p30_hpc', 'nf_fan_speed', 'nc_core_speed', 'ps_30_sta_press', 
    'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat', 
    'htbleed_enthalpy', 'w31_hpt_cool_bl', 'w32_lpt_cool_bl']

######    the training features has the columns to train to ### 
#######    the columns time_cycles and time_to_fail have been removed ##


####  The time cycles column may be used as an alternate y value to train to
y_cycles_to_fail = df1.cycles_to_fail
y_time_cycles = df1.time_cycles
####                                                                  #### 

##   view plots for the features that are to be used in df1   ######
for name in train_features:
    df1.plot.scatter( 'cycles_to_fail', name, alpha = .3)
    plt.show()



#### remove features that do not change at all for this dataset
for c in col:
    df1[c].describe()



## This will make the train test split for the model ####
y = y_time_cycles
X_features = df1[train_features]
Xtrain, Xtest, ytrain, ytest = train_test_split(X_features, y)


###   The train test split will include all engines for the start   ####  

#LINEAR: 
L_model = LinearRegression(fit_intercept=True)
L_model.fit(Xtrain.values, ytrain)
L_y_predicted = L_model.predict(Xtrain)



############ 
df1[col].describe()
raw.describe()

 ### Copy raw data to the df for work #####  
df = raw.copy()      # create a deep copy of dataframe  'raw' will not be touched in the processes
df.info()
df.head()
df.shape

