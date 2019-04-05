import pandas as pd


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
