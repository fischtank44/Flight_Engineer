import pandas as pd
from scipy import stats

#################################################################################
###               import data 

##### training #############

### Function to transform data frames
###  Add cycles to fail and y_failure (life remaining 100%)
def transform_dataframes_add_ys(data_list= [ ] , *args ):
# dataf = df1
    for df in data_list:
        max_cycles = []
        y_failure = []
        above_mean = []
        df_eng_max_cycles = []
        for e in range(1, max(df['unit'])+1 ):
            df_eng_max_cycles.append(max(df['time_cycles'][df['unit']==e]))
        for num in range(1, max(df['unit']) + 1):
            #print(num)
            max_cycles.append(max(df['time_cycles'][df['unit']==num] ) )
            # max_cycles
        cycles_to_fail = []
        for total in max_cycles:
            for cycle in range(total, 0, -1):
                # print(cycle)
                y_failure.append((cycle/total) )
                cycles_to_fail.append(cycle)
        # print(cycles_to_fail)
        # len(cycles_to_fail)
        # len(df)
        # len(y_failure)    
        # len(above_mean)
        # len(df)        
        df['cycles_to_fail'] = cycles_to_fail
        df['y_failure'] = y_failure
        c_mean = stats.describe(df_eng_max_cycles)[2]
        c_min = stats.describe(df_eng_max_cycles)[1][0]
        c_max = stats.describe(df_eng_max_cycles)[1][1]
        lower_third = ((c_max - c_min) / 3) + c_min
        middle_third = (( 2* (c_max - c_min) / 3 ) + c_min)
        # print(c_min, c_mean , c_max, lower_third, middle_third)
        above_mean_life = []
        for cycles in df_eng_max_cycles:
            # print(cycles)
            if cycles >= c_mean:
                for c in range(cycles):
                    above_mean_life.append( 1 )
            else:
                for c in range(cycles):
                    above_mean_life.append( 0 ) 
        # print(above_mean_life)
        # print(len(above_mean_life))
        # print(df.shape)
        df['above_mean_life'] = above_mean_life

        lower_third_list = []
        for cycles in df_eng_max_cycles:
            # print(cycles)
            if cycles <= lower_third:
                for c in range(cycles):
                    lower_third_list.append( 1 )
            else:
                for c in range(cycles):
                    lower_third_list.append( 0 )
        df['lower_third_life'] = lower_third_list 

        middle_third_list = []
        for cycles in df_eng_max_cycles:
            # print(cycles)
            if lower_third < cycles < middle_third:
                for c in range(cycles):
                    middle_third_list.append( 1 )
            else:
                for c in range(cycles):
                    middle_third_list.append( 0 )
        df['middle_third_life'] = middle_third_list

        upper_third_list = []
        for cycles in df_eng_max_cycles:
            # print(cycles)
            if cycles >= middle_third:
                for c in range(cycles):
                    upper_third_list.append( 1 )
            else:
                for c in range(cycles):
                    upper_third_list.append( 0 )
        df['upper_third_life'] = upper_third_list
