# python
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
#### add column to the end for logistic predictive model   ######
#### 

data_frames_to_transform = [df1, df2, df3 , df4]

def transform_dataframes_add_ys(data_list= [ ] , *args ):
# dataf = df1
    for df in data_list:
        max_cycles = []
        y_failure = []
        for num in range(1, max(df['unit']) + 1):
            #print(num)
            max_cycles.append(max(df['time_cycles'][df['unit']==num] ) )
            # max_cycles
        cycles_to_fail = []
        for total in max_cycles:
            for cycle in range(total, 0, -1):
                y_failure.append( 1-(cycle/total) )
                cycles_to_fail.append(cycle)
        # print(cycles_to_fail)
        len(cycles_to_fail)
        len(df)
        len(y_failure)            
        df['cycles_to_fail'] = cycles_to_fail
        df['y_failure'] = y_failure

# df1.cycles_to_fail

###   Transform all four dataframes   #######

transform_dataframes_add_ys(data_frames_to_transform)
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
test_engines = [4, 18, 19, 21, 28, 33, 42, 45, 46, 50, 61, 73, 74, 78, 82, 83, 84, 86, 92, 94]

train_engines = []
for num in range(1,101):
    if num not in test_engines:
        train_engines.append(num)
        #


train_engines = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 
    27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 47, 48, 49, 51, 52, 53, 54, 55, 
    56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 79, 80, 81, 85, 
    87, 88, 89, 90, 91, 93, 95, 96, 97, 98, 99, 100]

train_engines
test_engines



test_idx = df1['unit'].apply(lambda x: x in test_engines)
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

### Hold for future use  #######
# Xtrain, Xtest, ytrain, ytest = train_test_split(X_features, y, test_size = .2, random_state=137)
# Xtrain.shape
# Xtest.shape
# ytrain.shape
# ytest.shape


###   The train test split will include all engines for the start   ####  

# #LINEAR: 
# L_model = LinearRegression(fit_intercept=True)
# L_model.fit(X_features, ytrain)
# L_y_predicted = L_model.predict(X_features)


# L_y_predicted
# ############ 
# ######   Check the coefficients from the model 
# L_model.coef_
# print(list(zip(L_model.coef_, X_features)))


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
# #####   Plot the data from the first model and evaluate the residuals

# plt.scatter(L_y_predicted, ytrain, alpha = 0.1)
# plt.xlabel('y hat from training set')
# plt.ylabel( 'y values from training set')
# plt.show()
# ###






#### Second plot that will show the difference from actuals vs pred
# fig = plt.figure()
# fig, ax = plt.subplots(figsize=(15,15) )
# ax.plot(list(range(1, len(L_y_predicted) + 1)) , L_y_predicted, '.r', label='predicted')
# ax.plot(list(range(1, len(ytrain) + 1 )) , ytrain, '.b' , label='actual')
# plt.xlabel('Index of Value')
# plt.ylabel( 'Cycles to Fail')
# ax.legend()
# plt.show()

### First score from basic linear regression model   ####
# base_score = r2(ytrain, L_y_predicted)
# base_score
# linear_model_80_engine = base_score
# linear_model_80_engine

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
###    (ax, df, y, var_name,
for col in train_features:
    fig, ax = plt.subplots(figsize=(12, 3))
    plot_one_univariate(ax, df1, 'cycles_to_fail', col )
    ax.set_title("Evaluation of: " + str(col))
    plt.xlabel(col)
    plt.ylabel( 'Cycles to Fail')
    plt.show()

#### Begining of the linear spline transformation parameters    #######
# linear_spline_transformer = LinearSpline(knots=[10, 35, 50, 80, 130, 150, 200, 250, 300])

# linear_spline_transformer.transform(df1['cycles_to_fail']).head()

# cement_selector = ColumnSelector(name='cycles_to_fail')
# cement_column = cement_selector.transform('cycles_to_fail')
# linear_spline_transformer.transform(cement_column).head()

train_features

train_features = ['t24_lpc', 
't30_hpc', 
't50_lpt', 
'p30_hpc', 
'nf_fan_speed', 
'nc_core_speed', 
'ps_30_sta_press', 
'phi_fp_ps30', 
'nrf_cor_fan_sp', 
'nrc_core_sp', 
'bpr_bypass_rat', 
'htbleed_enthalpy', 
'w31_hpt_cool_bl', 
'w32_lpt_cool_bl']


t24_fit = Pipeline([
    ('t24_lpc', ColumnSelector(name='t24_lpc')),
    ('t24_lpc_spline', LinearSpline(knots=[641.5, 642,  642.5, 643.0 , 643.4, 644, 644.5]))
])



t30_fit = Pipeline([
    ('t30_hpc', ColumnSelector(name='t30_hpc')),
    ('t30_hpc_spline', LinearSpline(knots=[1573 , 1580, 1584, 1588, 1593, 1598 , 1610]))
])

t50_fit = Pipeline([
    ('t50_lpt', ColumnSelector(name='t50_lpt')),
    ('t50_lpt_spline', LinearSpline(knots=[1385, 1390, 1400, 1401, 1411, 1415, 1421, 1430, 1440]))
])


p30_fit = Pipeline([
    ('p30_hpc', ColumnSelector(name='p30_hpc')),
    ('p30_hpc_spline', LinearSpline(knots=[550, 552.2, 553.2, 554.8, 555, 555.5]))
])


nf_fan_fit = Pipeline([
    ('nf_fan_speed', ColumnSelector(name='nf_fan_speed')),
    ('nf_fan_speed_spline', LinearSpline(knots=[2387.9, 2388, 2388.1, 2388.15, 2388.2, 2388.3, 2388.4]))
])


nc_core_fit = Pipeline([
    ('nc_core_speed', ColumnSelector(name='nc_core_speed')),
    ('nc_core_speed_spline', LinearSpline(knots=[9030, 9040, 9060, 9070, 9080, 9090]))
])

ps_30_fit = Pipeline([
    ('ps_30_sta_press', ColumnSelector(name='ps_30_sta_press')),
    ('ps_30_sta_press_spline', LinearSpline(knots=[47, 47.2, 47.3, 47.45, 47.6, 47.8, 47.9, 48.25]))
])


phi_fp_fit = Pipeline([
    ('phi_fp_ps30', ColumnSelector(name='phi_fp_ps30')),
    ('phi_fp_ps30_spline', LinearSpline(knots=[519, 520, 520.4 , 521.2, 522, 522.4, 523]))
])


nrf_cor_fit = Pipeline([
    ('nrf_cor_fan_sp', ColumnSelector(name='nrf_cor_fan_sp')),
    ('nrf_cor_fan_sp_spline', LinearSpline(knots=[2387.9, 2388, 2388.6, 2388.2 , 2388.3, 2388.4]))
])

nrc_core_fit = Pipeline([
    ('nrc_core_sp', ColumnSelector(name='nrc_core_sp')),
    ('nrc_core_sp_spline', LinearSpline(knots=[8107.4 , 8117, 8127.5 , 8138.7 , 8149.4 , 8160 , 8171 , 8200 , 8250]))
])

bpr_bypass_fit = Pipeline([
    ('bpr_bypass_rat', ColumnSelector(name='bpr_bypass_rat')),
    ('bpr_bypass_rat_spline', LinearSpline(knots=[8.35 , 8.38, 8.41, 8.45, 8.49, 8.55]))
])


htbleed_fit = Pipeline([
    ('htbleed_enthalpy', ColumnSelector(name='htbleed_enthalpy')),
    ('htbleed_enthalpy_spline', LinearSpline(knots=[389, 390, 391, 392, 393, 394,395, 396, 397, 398, 399]))
])

w31_fit = Pipeline([
    ('w31_hpt_cool_bl', ColumnSelector(name='w31_hpt_cool_bl')),
    ('w31_hpt_cool_bl_spline', LinearSpline(knots=[38.2, 38.5, 38.7, 38.9, 39.1, 39.2]))
])

w32_fit = Pipeline([
    ('w32_lpt_cool_bl', ColumnSelector(name='w32_lpt_cool_bl')),
    ('w32_lpt_cool_bl_spline', LinearSpline(knots=[22.95, 23.14, 23.2,  23.32, 23.44]))
])



feature_pipeline = FeatureUnion([
    ('t24_lpc', t24_fit),
    ('t30_hpc', t30_fit),
    ('p30_hpc', p30_fit),
    ('t50_lpt', t50_fit),
    ('nf_fan_speed', nf_fan_fit),
    ('nc_core_speed', nc_core_fit),
    ('ps_30_sta_press', ps_30_fit),
    ('phi_fp_ps30', phi_fp_fit),
    ('nrf_cor_fan_sp', nrf_cor_fit),
    ('nrc_core_sp', nrc_core_fit),
    ("bpr_bypass_rat", bpr_bypass_fit),
    ("htbleed_enthalpy", htbleed_fit),
    ("w31_hpt_cool_bl", w31_fit),
    ("w32_lpt_cool_bl", w32_fit)
])




#### Build out the new dataframes with each knot   
#### Must use the 80 engine traing set !!!!!!!   

feature_pipeline.fit(df_new_train)
features = feature_pipeline.transform(df_new_train)


#####   


###    Fit model to the pipeline   #######
model = LinearRegression(fit_intercept=True)
model.fit(features.values, ytrain)


#### View the coefficients
display_coef(model, features.columns)



####  Make predictions against the training set
y_hat = model.predict(features.values)

####  Plot predictions from data against the actual values ########
x = list(range(1,320))
y = x
plt.scatter(y_hat, ytrain, alpha = 0.1, color='blue')
plt.plot(x, y, '-r', label='y=2x+1')
plt.title('First Pipline Predictions')
plt.xlabel('y hat from training set')
plt.ylabel( 'y actuals from training set')
plt.show()
###





#### Second plot that will show the difference from actuals vs pred for the pipeline model   ###### 

fig, ax = plt.subplots(figsize=(15,15) )
ax.plot(list(range(1, len(y_hat) + 1)) , y_hat, '.r', label='predicted')
ax.plot(list(range(1, len(ytrain) + 1 )) , ytrain, '.b' , label='actual')
plt.xlabel('Index of Value')
plt.ylabel( 'Cycles to Fail')
ax.legend()
plt.show()

##########################################


train_eng_max_cycles = []
for e in train_engines:
    train_eng_max_cycles.append(max(df1['time_cycles'][df1['unit']==e]))

# run

train_eng_max_cycles
 
 
    # #print(num)
    # max_cycles.append(max(df['time_cycles'][df['unit']==num] ) )




    # ax.set_title('Plot number {}'.format(i))


##### this is the plot of all 80 engines on a single chart

fig, axs = plt.subplots(8,10, figsize=(10,4))
ax.set_title("Spline Model of 80 Training Engines")
start_idx = 0
for idx, ax in enumerate(axs.flatten()):
# for idx, e in enumerate(train_engines):
    end_idx = start_idx + train_eng_max_cycles[idx]
    print(start_idx, end_idx, train_eng_max_cycles[idx], end_idx-start_idx)
    # fig, ax = plt.subplots(figsize=(15,15) )
    ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , y_hat[start_idx : end_idx], '.r', label='predicted')
    ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , ytrain[start_idx : end_idx] , '-b' , label='actual')
    ax.set_title("Engine # " + str(train_engines[idx]), size=6)
    # plt.tick_params(axis='both', which='major', labelsize=8)
    # plt.tick_params(axis='both', which='minor', labelsize=6)
    # plt.xticks(fontsize=8)      #, rotation=90)
    # plt.title('Engine #: ' + str(train_engines[idx]))
    # plt.xlabel('Index')
    # plt.ylabel( 'Cycles to Fail')
    # ax.legend()
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    start_idx = end_idx 
        # plt.show()


# plt.tight_layout()
plt.show()





#### Third plot that will show the difference from actuals vs pred for the pipeline model for each engine one by one  ###### 
start_idx = 0
for idx, e in enumerate(train_engines):
    end_idx = start_idx + train_eng_max_cycles[idx]
    print(start_idx, end_idx, train_eng_max_cycles[idx], end_idx-start_idx)
    fig, ax = plt.subplots(figsize=(15,15) )
    ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , y_hat[start_idx : end_idx], '.r', label='predicted')
    ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , ytrain[start_idx : end_idx] , '.b' , label='actual')
    plt.title('Engine #: ' + str(e))
    plt.xlabel('Index')
    plt.ylabel( 'Cycles to Fail')
    ax.legend()
    start_idx = end_idx 
    plt.show()



#### Score of the first model against the training set.  
## First score from basic linear regression model   ####
first_knot_model = r2(ytrain, y_hat)
first_knot_model

# first_knot_model
# 0.64194677350961

