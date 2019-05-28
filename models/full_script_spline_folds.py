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
from sklearn.model_selection import KFold, GroupKFold
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
# from export_linear_model import export_linear_model_to_txt
from engine_pipeline import fit_engine_pipeline

##################################

import os
os.getcwd()


np.random.seed(137)


#################################################################################
###               import data 

##### training #############

df1 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_01_fd.csv', sep= " " )
# df2 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_02_fd.csv', sep= ' ')
# df3 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_03_fd.csv', sep= ' ')
# df4 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/flight_engineer/enginedata/train_04_fd.csv', sep= ' ')

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


data_frames_to_transform = [df1]   # , df2, df3 , df4]
transform_dataframes_add_ys(data_frames_to_transform)
df = df1          #<----- #This is the dataframe to use for the model
target_variable = 'cycles_to_fail'  #   or 'y_failure'

##########################################################
# It will be a countdown of the total cycles for training set  ######
##  set dataf to dataframe name  ####
#### add column to the end for logistic predictive model   ######
#### 

# df1.cycles_to_fail


#####  End of data import file #######


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



#####   adjust the data frame to choose 20 % of the engines by unmber and 
#####   train to a sample of 80% by number and 20% saved for test data.

# test_engines_full = list(np.random.choice(range(1,101), 100, replace= False))
test_engines_full = [
    89, 97, 4, 26, 11, 74, 43, 100, 6, 80, 76, 79, 98, 
    3, 52, 24, 53, 58, 93, 27, 69, 29, 23, 91, 59, 65, 
    83, 13, 68, 5, 9, 63, 25, 46, 99, 1, 31, 88, 78, 7, 
    39, 62, 40, 48, 50, 15, 14, 35, 94, 81, 38, 90, 19, 75, 
    34, 84, 41, 95, 51, 47, 8, 60, 86, 44, 20, 21, 87, 16, 
    49, 36, 67, 37, 54, 61, 22, 77, 17, 2, 28, 71, 64, 33, 
    82, 57, 18, 55, 45, 10, 66, 12, 42, 70, 56, 92, 30, 32, 
    73, 96, 85, 72
    ]


fig, axs = plt.subplots(1, 5 , figsize=(10,10))
ax.set_title("R2 Values of Test Set")
# start_idx = 0
for idx, ax in enumerate(axs.flatten()):
    # for num in range(5):
    # print(idx)
    test_engines = test_engines_full[0+(20 * idx): 20 + (20 * idx)]
    # print(test_engines)
    train_engines = []
    for eng in range(1,101):
        if eng not in test_engines:
            train_engines.append(eng)
    # print(train_engines)
    test_idx = df['unit'].apply(lambda x: x in test_engines)
    train_idx = df['unit'].apply(lambda x: x in train_engines)
    test_list = list(test_idx)
    train_list = list(train_idx)
    df_new_test = df.iloc[test_list].copy()
    df_new_train = df.iloc[train_list].copy()
    y_train = df_new_train[target_variable]
    X_train_features = df_new_train[train_features]
    y_test = df_new_test[target_variable]
    X_test_features = df_new_test[train_features]
    feature_pipeline = fit_engine_pipeline()
    feature_pipeline.fit(df_new_train)
    features = feature_pipeline.transform(df_new_train)
    model = LinearRegression(fit_intercept=True)
    model.fit(features.values, np.log(y_train))
    y_hat_train = np.exp(model.predict(features.values) )
    print ("MSE for Training Set " + str(idx) + " "  + str(mse(y_train, y_hat_train)) )
    r2_values = r2_generator_last_n_cycles(y_train , y_hat_train, 350)
    # print(len(r2_values))
    # fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    ax.scatter(range(len(r2_values)+1, 1, -1) , r2_values , color = 'blue', label = 'training')
    ax.set_ylim(-2, 2500)
    ax.set_xlim(len(r2_values), 0)
    # ax.set_title('Mean Squared Error (Training): ' + str(idx+1))
    ax.set_xlabel('Cycles to Fail')
    ax.set_ylabel( 'MSE')
    feature_pipeline.fit(df_new_test)
    features = feature_pipeline.transform(df_new_test)
    y_hat = np.exp(model.predict(features.values) )
    # print (len(y_test) , len(y_hat) )
    # print (y_hat, type(y_hat))
    print ("MSE for Test Set " + str(idx) + " "  + str(mse(y_test, y_hat)))
    r2_values = r2_generator_last_n_cycles(y_test , y_hat, 400)
    # print(len(r2_values))
    # fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    ax.scatter(range(len(r2_values)+1, 1, -1) , r2_values ,  color = 'red', label = 'test') 
    ax.set_ylim(-2, 2500)
    ax.set_xlim(len(r2_values), 0)
    ax.set_title('Mean Squared Error: ' + str(idx+1))
    ax.set_xlabel('Cycles to Fail')
    ax.set_ylabel( 'MSE')
    plt.legend()



plt.show()   


########  Plot the r2 values as the number of cycles remaining approaches the end #######


# ##################################################
# [89, 97, 4, 26, 11, 74, 43, 100, 6, 80, 76, 79, 98, 3, 52, 24, 53, 58, 93, 27]
# # MSE for set 0 0.705671078930892
# [69, 29, 23, 91, 59, 65, 83, 13, 68, 5, 9, 63, 25, 46, 99, 1, 31, 88, 78, 7]
# MSE for set 1 0.677629932484811
# [39, 62, 40, 48, 50, 15, 14, 35, 94, 81, 38, 90, 19, 75, 34, 84, 41, 95, 51, 47]
# MSE for set 2 0.7424078838235062
# [8, 60, 86, 44, 20, 21, 87, 16, 49, 36, 67, 37, 54, 61, 22, 77, 17, 2, 28, 71]
# MSE for set 3 0.6802544775638418
# [64, 33, 82, 57, 18, 55, 45, 10, 66, 12, 42, 70, 56, 92, 30, 32, 73, 96, 85, 72]
# MSE for set 4 0.5632516095690059
##################################################






r2_for_last_n_cycles(y_hat , y, last_n=550)
r2_for_last_n_cycles(y_hat , y, last_n=100)
r2_for_last_n_cycles(y_hat , y, last_n=75)
r2_for_last_n_cycles(y_hat , y, last_n=50)
r2_for_last_n_cycles(y_hat , y, last_n=25)
r2_for_last_n_cycles(y_hat , y, last_n=15)
r2_for_last_n_cycles(y_hat , y, last_n=10)
r2_for_last_n_cycles(y_hat , y, last_n=5)




def r2_generator_last_n_cycles(y_hat , y_act, last_n=50):
    r_squared_vals = []
    for num in range(last_n, 0, -1):
        # print(num)
        ypred_n = []
        y_act_n = []
        for idx, cycle in enumerate(y_act):
            # print(num)
            if cycle <= num:
                # print(cycle, num, y_hat[idx])
                ypred_n.append(y_hat[idx])
                y_act_n.append(cycle)
        # print(ypred_n, y_act_n)
        r_squared_vals.append(r2(y_act_n , ypred_n) )
        # print(len(ytrain_n), len(y_act_n), len(r_squared_vals))
    return r_squared_vals

r2_generator_last_n_cycles( y_hat , y_test, 20)


y_hat
len(y_hat)
len(y_test)

y_hat = list(y_hat)
y_test = list(y_test)

for idx, val in enumerate(y_hat):
    print (r2(y_hat, y_test) )

r2(y_hat, y_test)
