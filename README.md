# Flight Engineer
Full spectrum aviation business study.

This is a multi-faceted analysis of an aviation business problem that grew from aircraft engine data available from NASA. [PCoE Datasets](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). This dataset is the publicly available: [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/publications/#turbofan).

## Engine Training Data: [Data Files](https://github.com/fischtank44/Engine_training_data/tree/master/Data_Files)

These files demonstrate analysis of data from a NASA study of turbofan engine failures. This is a front to back evaluation of the data beginning by using SQL to extract data, excel to clean data, and analyzing data in Tableau. Finally, two outputs are provided. The first is a predictive formula that attempts to provide an estimate of cycles remaining and the second is a real time assessment of the engines current health using visualizations all in Tableau.


## Data import:

The information from the training set needed to be imported to pandas for analysis. Each engine, the number of cycles, and data collected from each observation were included in the data. The information was relatively clean and organized; however, it did not have an included target variable.


## Training and test sets:

The training dataset was chosen by randomly sampling 80% of the engines that were run during the experiment. The remaining 20% of the engines were held in reserve for the test set. Some of the features were not used or where not predictive. The features for the model were evaluated based on their standard deviation and only those that had values more than  0.01 where included in the list of features to train to.   


## Target variable:

Since each engine was run to failure, the first target variable chosen was the exact number of cycles to failure for each engine. The maximum number of cycles that each engine ran varied from 128 – 362 cycles.

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/training_data_failure_distribution.png)

Thus it was possible for one engine to start with 250 life cycles remaining and another to begin its life with only 175 cycles. By setting the target variable to a countdown to 1 cycle remaining, it was possible to observe
the common trends shown by all of the engines as they approached failure.

The second target variable could be optimized for one of two hybrid values. The first could best be described as useful life remaining and the second as amount of useful life used up.

In either of these two hybrid cases the target for y is derived from the the number of the current cycle and the max number of cycles a specific engine operated to (for example 250 cycles). The target variable in this case would be set to start at:

![alt text](http://www.codecogs.com/gif.latex?\frac{1}{250} )

and it would end at:

![alt text](http://www.codecogs.com/gif.latex?\frac{250}{250} ).

Thus, it would count up from nearly 0 to 1. While interpreting this value as a probability of failure would be incorrect, it would behave in a manner that is similar to probability of failure. Values that are estimated to be at or near 1 would indicate that the engine was at or near the point of failure.

The same value can be subtracted from 1:

![alt text](http://www.codecogs.com/gif.latex?1-\frac{1}{250}=.996 )

and would then decrease from nearly 1 at the beginning of its life to 0 at the end.

Most importantly, it would allow for the compact form of a knot and spline model to be retained while providing values that closely approximate those derived from a soft probability classifier in the first case to a percent of useful life remaining in the second case.



## Models:
Linear regression with knots and splines.
Each feature that was included in the training data set was bootstrapped and plotted against cycles until failure. For example, the temperature recorded at location 50 at the low pressure turbine (t50_lpt) exit looked like this:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/t50_lpt_bs_spline_analysis.png)

Each feature values were then evaluated to determine where the trend showed a marked increase or decrease in slope. At each location a knot value was chosen. These knot locations where then fit to the original data-frame which was transformed using each of the knots as a new feature. This procedure was duplicated for each predictive variable shown in this series of graphs:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/all_features_cycles_to_fail.png)


### Linear regression predicting cycles to failure.
The plots of the first model indicated that the features contained data that was increasing/decreasing at a rate that was accelerating as the engines approached the end of their life cycle. As shown in this graph:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/pred_vs_actual_reg_regression.png)


### Linear regression predicting log transformed cycles to failure.
This was an indication that a transformation would be necessary. The first attempt was a natural log transformation of the target value (cycles to failure). The log variables started at values as high as 5.8916 (362 life cycles remaining), but all engines continued down to a value of 0 (1 life cycle remaining). The results where significantly improved as shown in this plot:   

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/training_cycles_to_fail.png)
