python

#######
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
from r_squared_funcs import (
    r2_for_last_n_cycles,
    r2_generator_last_n_cycles)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from basis_expansions.basis_expansions import NaturalCubicSpline
import random

#########################
###### Self made functions######
from r_squared_funcs import (
    r2_for_last_n_cycles,
    r2_generator_last_n_cycles)
from enginedatatransformer import transform_dataframes_add_ys
from plot_pred_vs_act import plot_many_predicteds_vs_actuals

##################################

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



small_features_list = [
    'time_cycles', 
    't24_lpc', 
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



#######      List of vaiables and features for model    #######


training_set = True
make_plots = False
data_frames_to_transform = [df1, df2, df3 , df4]
transform_dataframes_add_ys(data_frames_to_transform)
cols_to_use = small_features_list
df = df1          #<----- #This is the dataframe to use for the model
target_variable = 'cycles_to_fail'  #   or 'y_failure'

##########################################################
# It will be a countdown of the total cycles for training set  ######
##  set dataf to dataframe name  ####
#### add column to the end for logistic predictive model   ######
#### 

# df1.cycles_to_fail


#####  End of data import file #######


############  Start of data analysis   #############

## this will plot all columns to check for variation within the feature data
if make_plots==True:
    for name in col:
        df.plot.scatter( target_variable, name, alpha = .3)
        plt.show()
#



######     Several features appear to not be predictive  ######
#####     Scatter matrix using time cycles            ##### 
if make_plots==True:
    scatter_matrix = pd.scatter_matrix(df[cols_to_use], alpha=0.2, figsize=(20, 20), diagonal='kde')

if make_plots==True:
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 6, rotation = 90)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 6, rotation = 0)
        plt.show()



#####         Scatter matrix using cycles to fail        #####
if make_plots==True:
    scatter_matrix = pd.scatter_matrix(df[cols_to_use], alpha=0.2, figsize=(20, 20), diagonal='kde')




if make_plots==True:
    for ax in scatter_matrix.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize = 6, rotation = 90)
            ax.set_ylabel(ax.get_ylabel(), fontsize = 6, rotation = 0)
    plt.show()


#####                                                       ##### 


# #    view the description of each column 
# col = df.columns
# # col = train_features
# for c in col:
#   print (df[c].describe() ) 


# ### This will print only the standard deviation for each column
# col = df.columns
# for c in col:
#   print (df[c].describe()[2] ) 


### This will remove features based the standard deviation for each column
train_features = []
limit = .01
col = df.columns
for c in col:
  if (df[c].describe()[2] ) >= .01:
      train_features.append(c)

train_features
train_features = [ 
        'time_cycles', 
        't24_lpc', 
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
        'w32_lpt_cool_bl' 
]



######    the training features has the columns to train to ### 
#######    the columns time_cycles and time_to_fail have been removed ##


##   view plots for the features that are to be used in df   ######
if make_plots==True:
    for name in train_features:
        df.plot.scatter( target_variable, name, alpha = .3)
        plt.show()


#####   adjust the data frame to choose 20 % of the engines by unmber and 
#####   train to a sample of 80% by number and 20% saved for test data.
# test_engines = list(np.random.choice(range(1,101), 20, replace= False))
test_engines = [4, 18, 19, 21, 28, 33, 42, 45, 46, 50, 61, 73, 74, 78, 82, 83, 84, 86, 92, 94]


train_engines = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 
    27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 47, 48, 49, 51, 52, 53, 54, 55, 
    56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 79, 80, 81, 85, 
    87, 88, 89, 90, 91, 93, 95, 96, 97, 98, 99, 100]



# train_engines = []
# for num in range(1,101):
#     if num not in test_engines:
#         train_engines.append(num)
# #        #



train_engines
test_engines



############  Find index numbers for the training and test sets    ###########   
test_idx = df['unit'].apply(lambda x: x in test_engines)
train_idx = df['unit'].apply(lambda x: x in train_engines)
test_idx
train_idx

test_list = list(test_idx)
train_list = list(train_idx)





#### Create new data frames using seperate test and training engines   ##########
df_new_test = df.iloc[test_list].copy()
df_new_train = df.iloc[train_list].copy()
df_new_test.shape
df_new_train.shape



###### this will make a list of the max number of cycles for the training set of engines

train_eng_max_cycles = []
for e in train_engines:
    train_eng_max_cycles.append(max(df['time_cycles'][df['unit']==e]))


train_eng_max_cycles
stats.describe(train_eng_max_cycles)
#  DescribeResult(nobs=80, minmax=(128, 362), 
#  mean=203.4375, variance=2055.6922468354433, 
#  skewness=1.063155863408599, kurtosis=1.5047506637832253)



#######  the max number of cycles for the test set of engines  ########
test_eng_max_cycles = []
for e in test_engines:
    test_eng_max_cycles.append(max(df['time_cycles'][df['unit']==e]))

test_eng_max_cycles
stats.describe(test_eng_max_cycles)
# DescribeResult(nobs=20, minmax=(158, 341), 
# mean=217.8, variance=2469.326315789474, 
# skewness=0.8514362921848939, kurtosis=-0.005870492535239968)




# ### Show the max number of cycles for each unit in all of the sets. ######### 
# all_eng_max_cycles = []

# for e in range(1, max(df.unit)+1):
#     all_eng_max_cycles.append(max(df['time_cycles'][df['unit']==e]))

# all_eng_max_cycles


###########             Train to Cycles to Fail                ######################
###########@@@@@@@@    Toggle commments to change target   @@@@@########################


## This will make the train test split for this model ####
ytrain = df_new_train[target_variable]
X_train_features = df_new_train[train_features]
ytest = df_new_test[target_variable]
X_test_feaures = df_new_test[train_features]



# ###########                Train to y_failure (0-1)                ######################
# ## This will make the train test split for this model ####
# ytrain = df_new_train['y_failure']
# X_train_features = df_new_train[train_features]
# ytest = df_new_test['y_failure']
# X_test_feaures = df_new_test[train_features]




### Hold for future use  #######
# Xtrain, Xtest, ytrain, ytest = train_test_split(X_features, y, test_size = .2, random_state=137)
# Xtrain.shape
# Xtest.shape
# ytrain.shape
# ytest.shape




# #####   Plot the data from the first model and evaluate the residuals

# plt.scatter(L_y_predicted, ytrain, alpha = 0.1)
# plt.xlabel('y hat from training set')
# plt.ylabel( 'y values from training set')
# plt.show()
# ###



# Begin spline analysis of each significant feature
###### plot the full range of each engine against the cycles to fail
if make_plots==True:
    fig, axs = plt.subplots(3, 5, figsize=(14, 8))
    univariate_plot_names = df[train_features]                                     #columns[:-1]
    for name, ax in zip(univariate_plot_names, axs.flatten()):
        plot_univariate_smooth(ax,
                            df[target_variable],
                            df[name].values.reshape(-1, 1),
                            bootstrap=100)
        ax.set_title(name, fontsize=7)
    plt.show()



#### Plot each feature individually. 
###    (ax, df, y, var_name,
if make_plots==True:
    for col in train_features:
        fig, ax = plt.subplots(figsize=(12, 3))
        plot_one_univariate(ax, df, target_variable, col )
        ax.set_title("Evaluation of: " + str(col))
        plt.xlabel(col)
        plt.ylabel( 'Cycles to Fail')
        plt.show()


train_features

cycle_fit = Pipeline([
    ('time_cycles', ColumnSelector(name='time_cycles')),
    ('time_cycles_spline', LinearSpline(knots=[25, 50, 75, 120, 175 , 220, 240, 260, 280, 300]))
])

t24_fit = Pipeline([
    ('t24_lpc', ColumnSelector(name='t24_lpc')),
    ('t24_lpc_spline', LinearSpline(knots=[641.5, 642,  642.5, 643.0 , 643.4, 644]))
])

t30_fit = Pipeline([
    ('t30_hpc', ColumnSelector(name='t30_hpc')),
    ('t30_hpc_spline', LinearSpline(knots=[ 1580, 1584, 1588, 1593, 1598 , 1610]))
])

t50_fit = Pipeline([
    ('t50_lpt', ColumnSelector(name='t50_lpt')),
    ('t50_lpt_spline', LinearSpline(knots=[1385, 1390, 1400, 1401, 1411, 1415, 1421, 1430, 1440]))
])

p30_fit = Pipeline([
    ('p30_hpc', ColumnSelector(name='p30_hpc')),
    ('p30_hpc_spline', LinearSpline(knots=[ 552.2, 553.2, 554.8, 555, 555.5]))
])

nf_fan_fit = Pipeline([
    ('nf_fan_speed', ColumnSelector(name='nf_fan_speed')),
    ('nf_fan_speed_spline', LinearSpline(knots=[2387.9, 2388, 2388.1, 2388.15, 2388.2, 2388.3]))
])

nc_core_fit = Pipeline([
    ('nc_core_speed', ColumnSelector(name='nc_core_speed')),
    ('nc_core_speed_spline', LinearSpline(knots=[ 9040, 9060, 9070, 9080, 9090]))
])

ps_30_fit = Pipeline([
    ('ps_30_sta_press', ColumnSelector(name='ps_30_sta_press')),
    ('ps_30_sta_press_spline', LinearSpline(knots=[47, 47.2, 47.3, 47.45, 47.6, 47.8, 47.9]))
])

phi_fp_fit = Pipeline([
    ('phi_fp_ps30', ColumnSelector(name='phi_fp_ps30')),
    ('phi_fp_ps30_spline', LinearSpline(knots=[ 520, 520.4 , 521.2, 522, 522.4, 523]))
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
    ('bpr_bypass_rat_spline', LinearSpline(knots=[8.35 , 8.38, 8.41, 8.45, 8.49]))
])


htbleed_fit = Pipeline([
    ('htbleed_enthalpy', ColumnSelector(name='htbleed_enthalpy')),
    ('htbleed_enthalpy_spline', LinearSpline(knots=[389, 390, 391, 392, 393, 394,395, 396, 397, 398, 399]))
])

w31_fit = Pipeline([
    ('w31_hpt_cool_bl', ColumnSelector(name='w31_hpt_cool_bl')),
    ('w31_hpt_cool_bl_spline', LinearSpline(knots=[38.5, 38.7, 38.9, 39.1, 39.2]))
])

w32_fit = Pipeline([
    ('w32_lpt_cool_bl', ColumnSelector(name='w32_lpt_cool_bl')),
    ('w32_lpt_cool_bl_spline', LinearSpline(knots=[ 23.14, 23.2,  23.32, 23.44]))
])



feature_pipeline = FeatureUnion([
    ('time_cycles', cycle_fit),
    ('t24_lpc', t24_fit),
    ('t30_hpc', t30_fit),
    ('p30_hpc', p30_fit),
    ('t50_lpt', t50_fit),
    ('nf_fan_speed', nf_fan_fit),
    # ('nc_core_speed', nc_core_fit),
    ('ps_30_sta_press', ps_30_fit),
    ('phi_fp_ps30', phi_fp_fit),
    ('nrf_cor_fan_sp', nrf_cor_fit),
    # ('nrc_core_sp', nrc_core_fit),
    ("bpr_bypass_rat", bpr_bypass_fit),
    ("htbleed_enthalpy", htbleed_fit),
    ("w31_hpt_cool_bl", w31_fit),
    ("w32_lpt_cool_bl", w32_fit)
])






#### Build out the new dataframes with each knot  
# 
# 
# ####   Full transformation of data frame  using pipline ##########
# feature_pipeline.fit(df)
# features = feature_pipeline.transform(df)

#########################################################


  #### Must use the 80 engine traing set !!!!!!!   
if training_set == True:
    feature_pipeline.fit(df_new_train)
    features = feature_pipeline.transform(df_new_train)

####################################################

  
# #### Build out the new dataframes with each knot   
# #### Must use the 20 engine test set !!!!!!!   
if training_set == False:
    feature_pipeline.fit(df_new_test)
    features = feature_pipeline.transform(df_new_test)

# ##################################################



###    Fit train model to the pipeline   #######
if training_set == True:
    model = LinearRegression(fit_intercept=True)
    model.fit(features.values, np.log(ytrain)) # <---- note: the np.log transformation
####  Make predictions against the training set
    y_hat = model.predict(features.values)
    y_hat = np.exp(y_hat)                ## <----- note: the exp to transform back
len(y_hat)
len(ytrain)
len(features)





###    Fit test model to the pipeline   #######
if training_set == False:
    model = LinearRegression(fit_intercept=True)
    model.fit(features.values, np.log(ytest)) # <---- note: the np.log transformation
####  Make predictions against the test set
    y_hat = model.predict(features.values)
    y_hat = np.exp(y_hat)                ## <----- note: the exp to transform back

len(y_hat)
len(ytest)
len(features)





####  Plot predictions from data against the actual values ########
if make_plots==True:
    x = list(range( 1,360))
    y = x
    plt.scatter(y_hat, ytrain, alpha = 0.1, color='blue')
    plt.plot(x, y, '-r', label='y=2x+1')
    plt.title('Pipline Predictions with log(y)')
    plt.xlabel('$\hat {y}$ from training set')
    plt.ylabel( 'y actuals from training set')
    plt.xlim(360,1)
    plt.show()
###



#### Second plot that will show the difference from actuals vs pred for the pipeline model   ###### 
if make_plots==True:
    fig, ax = plt.subplots(figsize=(15,15) )
    ax.plot(list(range(1, len(y_hat) + 1)) , y_hat, '.r', label='predicted')
    ax.plot(list(range(1, len(ytrain) + 1 )) , ytrain, '.b' , label='actual')
    plt.xlabel('Index of Value')
    plt.ylabel( 'Cycles to Fail')
    ax.legend()
    plt.show()

##########################################



##### this is the plot of all 80 engines on a single chart  #####
#####        Training Set Data Plots        ######### 

if make_plots==True and training_set==True:
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
        ax.set_ylim([0,350])
        ax.set_xlim([350, 0])
        start_idx = end_idx 
    plt.show()

              ################################


##### Test Set of data    ###############################
##### this is the plot of all 20 test engines on a single chart

if make_plots==True and training_set==False:
    fig, axs = plt.subplots(4, 5 , figsize=(10,4))
    ax.set_title("Spline Model of 20 Test Engines")
    start_idx = 0
    for idx, ax in enumerate(axs.flatten()):
    # for idx, e in enumerate(train_engines):
        end_idx = start_idx + test_eng_max_cycles[idx]
        print(start_idx, end_idx, test_eng_max_cycles[idx], end_idx-start_idx)
        # fig, ax = plt.subplots(figsize=(15,15) )
        # ax.plot(y_hat[start_idx : end_idx], list(range(train_eng_max_cycles[idx], 0, -1)), '.r', label='predicted')
        # ax.plot(ytrain[start_idx : end_idx] , list(range(train_eng_max_cycles[idx], 0, -1)) , '-b' , label='actual')
        ax.plot(list(range(test_eng_max_cycles[idx], 0, -1)) , y_hat[start_idx : end_idx], '.r', label='predicted')
        ax.plot(list(range(test_eng_max_cycles[idx], 0, -1)) , ytest[start_idx : end_idx] , '-b' , label='actual')
        ax.set_title("Engine # " + str(test_engines[idx]), size=6)
        # plt.tick_params(axis='both', which='major', labelsize=8)
        # plt.tick_params(axis='both', which='minor', labelsize=6)
        # plt.xticks(fontsize=8)      #, rotation=90)
        # plt.title('Engine #: ' + str(train_engines[idx]))
        # plt.xlabel('Index')
        # plt.ylabel( 'Cycles to Fail')
        # ax.legend()
        ax.set_ylim( 0  , 350 )
        ax.set_xlim(350 ,  0)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        start_idx = end_idx 
    plt.show()




test_eng_max_cycles
train_eng_max_cycles




np.mean(train_eng_max_cycles)
np.mean(test_eng_max_cycles)


#### Third plot that will show the difference from actuals vs pred for
# #   the pipeline model for each engine one by one  ###### 
if make_plots==True and training_set==True:
    start_idx = 0
    for idx, e in enumerate(train_engines):
        end_idx = start_idx + train_eng_max_cycles[idx]
        print(start_idx, end_idx, train_eng_max_cycles[idx], end_idx-start_idx)
        fig, ax = plt.subplots(figsize=(15,15) )
        ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , y_hat[start_idx : end_idx], '.r', label='predicted')
        ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , ytrain[start_idx : end_idx] , '.b' , label='actual')
        plt.title('Engine #: ' + str(e))
        plt.xlabel('Cycles to Fail')
        plt.ylabel( 'Cycles Used')
        plt.axvline(stats.describe(train_eng_max_cycles)[1][0], color='r', label='min' )
        plt.axvline(stats.describe(train_eng_max_cycles)[2], color='g' , label='avg' )
        plt.axvline(stats.describe(train_eng_max_cycles)[1][1], color='b' , label='max' )
        
        ax.legend()
        start_idx = end_idx 
        plt.show()


# #   the pipeline model for each engine one by one  ###### 
if make_plots==True and training_set==False:
    start_idx = 0
    for idx, e in enumerate(train_engines):
        end_idx = start_idx + train_eng_max_cycles[idx]
        print(start_idx, end_idx, train_eng_max_cycles[idx], end_idx-start_idx)
        fig, ax = plt.subplots(figsize=(15,15) )
        ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , y_hat[start_idx : end_idx], '.r', label='predicted')
        ax.plot(list(range(train_eng_max_cycles[idx], 0, -1)) , ytrain[start_idx : end_idx] , '.b' , label='actual')
        plt.title('Engine #: ' + str(e))
        plt.xlabel('Cycles to Fail')
        plt.ylabel( 'Cycles Used')
        plt.axvline(stats.describe(train_eng_max_cycles)[1][0], color='r', label='min' )
        plt.axvline(stats.describe(train_eng_max_cycles)[2], color='g' , label='avg' )
        plt.axvline(stats.describe(train_eng_max_cycles)[1][1], color='b' , label='max' )
        
        ax.legend()
        start_idx = end_idx 
        plt.show()







### This will plot the final estimations vs the actual data


# y_hat = model.predict(df_new_train.values )




if make_plots==True and training_set==True:
    fig, axs = plot_many_predicteds_vs_actuals(train_features, y_hat)
    # fig.tight_layout()df
    plt.show()

if make_plots==True and training_set==False:
    fig, axs = plot_many_predicteds_vs_actuals(train_features, y_hat)
    # fig.tight_layout()df
    plt.show()


len(y_hat)
len(train_features)
train_features
len(df_new_train)

##########################    Scoreing Section   ###############



#### Score of the first model against the training set.  
## First score from basic linear regression model   ####
log_knot_model = r2(ytrain, y_hat)
log_knot_model
# time_knot_model
# first_knot_model
# 0.64194677350961
# 0.7396060171044228
# log_knot_model
# 0.7272227017732488
#log_knot_model
# 0.7273228097635444




##### R-squared for the last n number of observations  #####
#
ytrain
y_hat

r2_for_last_n_cycles(y_hat , ytrain, last_n=150)
r2_for_last_n_cycles(y_hat , ytrain, last_n=100)
r2_for_last_n_cycles(y_hat , ytrain, last_n=75)
r2_for_last_n_cycles(y_hat , ytrain, last_n=50)
r2_for_last_n_cycles(y_hat , ytrain, last_n=25)
r2_for_last_n_cycles(y_hat , ytrain, last_n=15)
r2_for_last_n_cycles(y_hat , ytrain, last_n=10)
r2_for_last_n_cycles(y_hat , ytrain, last_n=5)

###################   Make a list of r squared values for plotting   ##########

if training_set == True:
    r2_values = r2_generator_last_n_cycles(y_hat , ytrain, 200)


if training_set == False:
    r2_values = r2_generator_last_n_cycles(y_hat , ytest, 200)

########  Plot the r2 values as the number of cycles remaining approaches the end #######

##### plot the full against the cycles to fail
if make_plots == True:
    fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    ax.scatter(range(len(r2_values)+1, 1, -1) , r2_values)
    plt.ylim(-2, 1)
    plt.xlim(len(r2_values), 0)
    plt.title('R Squared')
    plt.xlabel('Cycles to Fail')
    plt.ylabel( 'R Squared Value')
    plt.show()

### Plot of r-squared as the number of observations approaches 1  #########







####################################################
####   Test for full transformation of data frame  ##########






############################################################################
# This creates a list of models, one for each bootstrap sample.


feature_pipeline.fit(df_new_train)
features_f = feature_pipeline.transform(df_new_train)


model = LinearRegression(fit_intercept=True)
model.fit(features_f.values, np.log(df_new_train[target_variable])) 

cols_to_use = [
    'time_cycles', 
    't24_lpc', 
    't30_hpc', 
    't50_lpt', 
    'p30_hpc', 
    'nf_fan_speed', 
    # 'nc_core_speed', 
    'ps_30_sta_press', 
    'phi_fp_ps30', 
    'nrf_cor_fan_sp', 
    # 'nrc_core_sp', 
    'bpr_bypass_rat', 
    'htbleed_enthalpy', 
    'w31_hpt_cool_bl', 
    'w32_lpt_cool_bl']


# feature_pipeline.fit(df)
# features = feature_pipeline.transform(df)

models = bootstrap_train(
    LinearRegression, 
    features_f.values, 
    np.log(df_new_train[target_variable].values),
    bootstraps=500,
    fit_intercept=True
)


# fig, axs = plot_bootstrap_coefs(models, features.columns, n_col=4)
# plt.show()


fig, axs = plot_partial_dependences(
     model, 
     X=df_new_train,
     var_names=cols_to_use,
     pipeline=feature_pipeline,
     bootstrap_models=models,
     y=None#np.log(df[target_variable]).values  
     )
# fig.tight_layout()


plt.show()



df_new_train.head()
df_new_train.shape



