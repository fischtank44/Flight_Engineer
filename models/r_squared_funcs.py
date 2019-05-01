import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse


def r2_for_last_n_cycles(y_act , y_hat, last_n=50):
    ypred_n = []
    y_act_n = []
    for idx, cycle in enumerate(y_act):
        # print(cycle)
        if cycle <= last_n:
            ypred_n.append(y_hat[idx])
            y_act_n.append(cycle)
    # print(len(ytrain_n))
    # print(len(y_act_n))
    return ("The r-squared for the last %s cycles is: " + str(mse(y_act_n, ypred_n) )) % last_n



###################   Make a list of r squared values for plotting   ##########

def r2_generator_last_n_cycles(y_act , y_hat, last_n=50):
    r_squared_vals = []
    # print (y_hat)
    for n in range(last_n, 0, -1):
        # print(n)
        ypred_n = []
        y_act_n = []
        for idx, cycle in enumerate(y_act):
            # print(n)
            if cycle <= n:
                # print(cycle, n)
                ypred_n.append(float(y_hat[idx]) )
                y_act_n.append(cycle)
        # print(ytrain_n, y_act_n)
        # print( len(y_act_n) , len(ypred_n)) 
        r_squared_vals.append(mse(y_act_n , ypred_n) )
        # print(len(ytrain_n), len(y_act_n), len(r_squared_vals))
    return r_squared_vals

