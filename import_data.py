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

#################################################################################
###               import data 

##### training #############

df1 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_01_fd.csv', sep= " " )
df2 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_02_fd.csv', sep= ' ')
df3 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_03_fd.csv', sep= ' ')
df4 = pd.read_csv('/home/superstinky/Seattle_g89/final_project_data/enginedata/train_04_fd.csv', sep= ' ')

################   This will add a column for the y value which will be the number of cycles until the engine fails.
# It will be a countdown of the total cycles 

max_cycles = []
for num in range(1, max(df1['unit']) + 1):
  print(num)
  max_cycles.append(max(df1['time_cycles'][df1['unit']==num] ) )
  max_cycles
  len(max_cycles)

############################

#    view the description of each column 
col = df1.columns
for c in col:
  print (df1[c].describe() ) 


### This will print only the standard deviation for each column
col = df1.columns
for c in col:
  print (df1[c].describe()[2] ) 

## this will plot all columns to check for any variation in the data
for name in col:
    df1.plot.scatter( 'cycles_to_fail', name, alpha = .3)
    plt.show()


# use column discribe out how remove the columns that do not change #### 




col = ['unit', 'time_cycles', 'op_set_1', 'op_set_2', 'op_set_3', 't2_Inlet',
       't24_lpc', 't30_hpc', 't50_lpt', 'p2_fip', 'p15_pby', 'p30_hpc',
       'nf_fan_speed', 'nc_core_speed', 'epr_p50_p2', 'ps_30_sta_press',
       'phi_fp_ps30', 'nrf_cor_fan_sp', 'nrc_core_sp', 'bpr_bypass_rat',
       'far_b_air_rat', 'htbleed_enthalpy', 'nf_dmd_dem_fan_sp', 'pcn_fr_dmd',
       'w31_hpt_cool_bl', 'w32_lpt_cool_bl']

#### remove features that do not change at all for this dataset
for c in col:
    df1[c].describe()


############ 
df1[col].describe()
raw.describe()

 ### Copy raw data to the df for work #####  
df = raw.copy()      # create a deep copy of dataframe  'raw' will not be touched in the processes
df.info()
df.head()
df.shape

