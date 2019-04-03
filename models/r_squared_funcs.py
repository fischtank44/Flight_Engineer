import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as r2


def r2_for_last_n_cycles(y_hat , y_act, last_n=50):
    ytrain_n = []
    y_act_n = []
    for idx, cycle in enumerate(y_act):
        # print(cycle)
        if cycle <= last_n:
            ytrain_n.append(cycle)
            y_act_n.append(y_hat[idx])
    # print(len(ytrain_n))
    # print(len(y_act_n))
    return ("The r-squared for the last %s cycles is: " + str(r2(ytrain_n, y_act_n) )) % last_n



###################   Make a list of r squared values for plotting   ##########

def r2_generator_last_n_cycles(y_hat , y_act, last_n=50):
    r_squared_vals = []
    for num in range(last_n, 0, -1):
        # print(num)
        ytrain_n = []
        y_act_n = []
        for idx, cycle in enumerate(y_act):
            # print(num)
            if cycle <= num:
                # print(cycle, num)
                ytrain_n.append(cycle +.0000000000001)
                y_act_n.append(y_hat[idx])
        # print(ytrain_n, y_act_n)
        r_squared_vals.append(r2(ytrain_n, y_act_n) )
        # print(len(ytrain_n), len(y_act_n), len(r_squared_vals))
    return r_squared_vals

