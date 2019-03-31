# python
import numpy as np
import pandas as pd
# from pandas.plotting import scatter_matrix

from regression_tools.dftransformers import (
    ColumnSelector, Identity,
    FeatureUnion, MapFeature,
    StandardScaler)

from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
from math import ceil
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from basis_expansions.basis_expansions import NaturalCubicSpline


import os
os.getcwd()


np.random.seed(137)


#################################################################################
###               import data 

##### training #############

df1 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_01_fd.csv', sep= " " )
df2 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_02_fd.csv', sep= ' ')
df3 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_03_fd.csv', sep= ' ')
df4 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_04_fd.csv', sep= ' ')

################   This will add a column for the y value which will be the number of cycles until the engine fails.
# It will be a countdown of the total cycles for training set  ######

##  set dataf to dataframe name  ####
dataf = df4
max_cycles = []
for num in range(1, max(dataf['unit']) + 1):
#   print(num)
    max_cycles.append(max(dataf['time_cycles'][dataf['unit']==num] ) )
#   len(max_cycles)
    cycles_to_fail = []
    for total in max_cycles:
        for cycle in range(total, 0, -1):
            cycles_to_fail.append(cycle)

    # print(cycles_to_fail)
    # len(cycles_to_fail)
    # len(df1)
dataf['cycles_to_fail'] = cycles_to_fail
# dataf[dataf['unit']==1]
### add the cycles to fail on to the original data frame. #####
dataf = df4
df4.cycles_to_fail

############################

# use column discribe out how remove the columns that do not change #### 


col = ['unit', 'time_cycles', 'op_set_1', 'op_set_2', 'op_set_3', 't2_Inlet',
       't24_lpc', 't30_hpc', 't50_lpt', 'p2_fip', 'p15_pby', 'p30_hpc',
       'nf_fan_speed', 'nc_core_speed', 'epr_p50_p2', 'ps_30_sta_press',
       'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat',
       'far_b_air_rat', 'htbleed_enthalpy', 'nf_dmd_dem_fan_sp', 'pcn_fr_dmd',
       'w31_hpt_cool_bl', 'w32_lpt_cool_bl', 'cycles_to_fail']

#####  End of data import file #######


############  Start of data analysis   #############



##### plot scatter plot for all features    #####
pd.tools.plotting.scatter_matrix(df1[col], figsize=(10, 10), s=100)
plt.show()

######     Several features are not predictive  ######

#   limit the features that are in the model #####
small_features_list = ['time_cycles', 't24_lpc', 't30_hpc', 't50_lpt', 
    'p30_hpc', 'nf_fan_speed', 'nc_core_speed', 'ps_30_sta_press', 
    'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat', 
    'htbleed_enthalpy', 'w31_hpt_cool_bl', 'w32_lpt_cool_bl' ]

pd.tools.plotting.scatter_matrix(df1[small_features_list], figsize=(10, 10), s=100)
plt.show()

#####                                                       ##### 

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
Xtrain.shape
Xtest.shape
ytrain.shape
ytest.shape


###   The train test split will include all engines for the start   ####  

#LINEAR: 
L_model = LinearRegression(fit_intercept=True)
L_model.fit(Xtrain.values, ytrain)
L_y_predicted = L_model.predict(Xtrain)


L_y_predicted
############ 


#####   Plot the data from the first model and evaluate the residuals

plt.scatter(L_y_predicted, ytrain, alpha = 0.1)
plt.xlabel('y hat from training set')
plt.ylabel( 'y values from training set')
plt.show()


#### Second plot that will show the difference from actuals vs pred
# fig = plt.figure()
fig, ax = plt.subplots(figsize=(15,15) )
ax.plot(list(range(1, len(L_y_predicted) + 1)) , L_y_predicted, '.r', label='predicted')
ax.plot(list(range(1, len(ytrain) + 1 )) , ytrain, '.b' , label='actual')
plt.xlabel('Index of Value')
plt.ylabel( 'Cycles to Fail')
ax.legend()
plt.show()

### First score from basic linear regression model   ####
base_score = r2(ytrain, L_y_predicted)


