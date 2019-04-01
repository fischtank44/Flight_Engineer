# python
import numpy as np
import pandas as pd
# from pandas.plotting import scatter_matrix

from regression_tools.dftransformers import (
    ColumnSelector, Identity,
    FeatureUnion, MapFeature,
    StandardScaler)
from plot_univariate import plot_one_univariate
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
import random

import os
os.getcwd()


np.random.seed(137)


#################################################################################
###               import data 

##### training #############

df1 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_01_fd.csv', sep= " " )
df2 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_02_fd.csv', sep= ' ')
df3 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_03_fd.csv', sep= ' ')
df4 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_04_fd.csv', sep= ' ')

################   This will add a column for the y value which will be the number of cycles until the engine fails.
# It will be a countdown of the total cycles for training set  ######

##  set dataf to dataframe name  ####
dataf = df1
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
df1 = dataf
# df1.cycles_to_fail

############################

# use column discribe out how remove the columns that do not change #### 

col = df1.columns
col = ['unit', 'time_cycles', 'op_set_1', 'op_set_2', 'op_set_3', 't2_Inlet',
       't24_lpc', 't30_hpc', 't50_lpt', 'p2_fip', 'p15_pby', 'p30_hpc',
       'nf_fan_speed', 'nc_core_speed', 'epr_p50_p2', 'ps_30_sta_press',
       'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat',
       'far_b_air_rat', 'htbleed_enthalpy', 'nf_dmd_dem_fan_sp', 'pcn_fr_dmd',
       'w31_hpt_cool_bl', 'w32_lpt_cool_bl', 'cycles_to_fail']

#####  End of data import file #######


############  Start of data analysis   #############



##### plot scatter plot for all features    #####
pd.tools.plotting.scatter_matrix(df1[col], figsize=(10, 10), s=100, alpha=.3)
plt.show()




## this will plot all columns to check for variation within the feature data
for name in col:
    df1.plot.scatter( 'cycles_to_fail', name, alpha = .3)
    plt.show()


######     Several features appear to not be predictive  ######

#   limit the features that are in the model scatter plot #####
small_features_list = ['time_cycles', 't24_lpc', 't30_hpc', 't50_lpt', 
    'p30_hpc', 'nf_fan_speed', 'nc_core_speed', 'ps_30_sta_press', 
    'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat', 
    'htbleed_enthalpy', 'w31_hpt_cool_bl', 'w32_lpt_cool_bl' ]

#####     Below is the cycles to fail columns        ##### 

_ = scatter_matrix(df1[small_features_list], alpha=0.2, figsize=(20, 20), diagonal='kde')
plt.show()


#   limit the features that are in the model scatter plot #####
small_features_list = ['cycles_to_fail' , 't24_lpc', 't30_hpc', 't50_lpt', 
    'p30_hpc', 'nf_fan_speed', 'nc_core_speed', 'ps_30_sta_press', 
    'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat', 
    'htbleed_enthalpy', 'w31_hpt_cool_bl', 'w32_lpt_cool_bl' ]

_ = scatter_matrix(df1[small_features_list], alpha=0.2, figsize=(20, 20), diagonal='kde')
plt.show()

#####                                                       ##### 


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

#### List of features to train the model to  #######    ### remove 'unit'
train_features = [ 't24_lpc', 't30_hpc', 't50_lpt', 
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

#####   adjust the data frame to choose 20 % of the engines by unmber and 
#####   train to a sample of 80% by number and 20% saved for test data.
# engines = list(np.random.choice(range(1,101), 20, replace= False))
engines = [4, 18, 19, 21, 28, 33, 42, 45, 46, 50, 61, 73, 74, 78, 82, 83, 84, 86, 92, 94]

train_engines = []
for num in range(1,101):
    if num not in engines:
        train_engines.append(num)
train_engines


test_idx = df1['unit'].apply(lambda x: x in engines)
train_idx = df1['unit'].apply(lambda x: x in train_engines)
test_idx
train_idx


type(test_idx)
type(train_idx)
test_list = list(test_idx)
train_list = list(train_idx)



df_new_test = df1.iloc[test_list].copy()
df_new_train = df1.iloc[train_list].copy()
df_new_test.shape
df_new_train.shape


## This will make the train test split for the model ####
ytrain = df_new_train['cycles_to_fail']
X_features = df_new_train[train_features]


Xtrain, Xtest, ytrain, ytest = train_test_split(X_features, y, test_size = .2, random_state=137)
Xtrain.shape
Xtest.shape
ytrain.shape
ytest.shape


###   The train test split will include all engines for the start   ####  

#LINEAR: 
L_model = LinearRegression(fit_intercept=True)
L_model.fit(X_features, ytrain)
L_y_predicted = L_model.predict(X_features)


L_y_predicted
############ 
######   Check the coefficients from the model 
L_model.coef_
print(list(zip(L_model.coef_, X_features)))


##### Model from old train/test split
# [(0.2098130774662108, 'unit'), (-7.173759447981604, 't24_lpc'), 
# (-0.42305195925658207, 't30_hpc'), (-0.7441639445488603, 't50_lpt'), 
# (7.61219378587503, 'p30_hpc'), (-12.147203483784747, 'nf_fan_speed'), 
# (-0.3844533247091928, 'nc_core_speed'), (-34.641657728829905, 'ps_30_sta_press'),
#  (11.105368284298036, 'phi_fp_ps30'), (-4.474447225499914, 'nrf_cor_fan_sp'), 
#  (-0.20542361139388693, 'nrc_core_sp'), (-126.19522472669553, 'bpr_bypass_rat'), 
#  (-1.9171623154921535, 'htbleed_enthalpy'), (22.12461560626438, 'w31_hpt_cool_bl'),
#  (42.47336192785645, 'w32_lpt_cool_bl')]
#
#  Model from new 80 engine 20 test train/test split
# #print(list(zip(L_model.coef_, X_features)))
# [(-7.9993983227825884, 't24_lpc'), (-0.40343998913641343, 't30_hpc'), 
# (-0.858069141166363, 't50_lpt'), (7.118138412200282, 'p30_hpc'), 
# (-26.53526438485433, 'nf_fan_speed'), (-0.28820253265246504, 'nc_core_speed'), 
# (-38.13957596837547, 'ps_30_sta_press'), (9.984072018801038, 'phi_fp_ps30'), 
# (-21.747334830714323, 'nrf_cor_fan_sp'), (-0.28742611769798204, 'nrc_core_sp'), 
# (-101.5927346354093, 'bpr_bypass_rat'), (-1.6264557877934611, 'htbleed_enthalpy'),
#  (19.17595070701376, 'w31_hpt_cool_bl'), (42.100133123738566, 'w32_lpt_cool_bl')]
#
#
#
#
#####   Plot the data from the first model and evaluate the residuals

plt.scatter(L_y_predicted, ytrain, alpha = 0.1)
plt.xlabel('y hat from training set')
plt.ylabel( 'y values from training set')
plt.show()
###






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
base_score
linear_model_80_engine = base_score
linear_model_80_engine

#####  score of model no tuning trained to time cycles to go
##  0.5302416225409862

#### score of model with no tuning trained to cycles remaining 
##  0.5302416225409862
##
### There is no difference between the two which makes sense.

####  Linear model 80 engine split 
# linear_model_80_engine
# 0.6004573742141459



# Begin spline analysis of each significant feature
# plot the full range of each engine against the cycles to fail
fig, axs = plt.subplots(3, 5, figsize=(14, 8))
univariate_plot_names = df1[train_features]                                     #columns[:-1]

for name, ax in zip(univariate_plot_names, axs.flatten()):
    plot_univariate_smooth(ax,
                           df1['cycles_to_fail'],
                           df1[name].values.reshape(-1, 1),
                           bootstrap=100)
    ax.set_title(name)
plt.show()



#### Plot each feature individually. 
