# Flight Engineer [Handout](https://github.com/fischtank44/Flight_Engineer/raw/master/Flight-engineer-writeup.pdf)
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

and would then decrease from nearly 1 at the beginning of its life to 0 at the end. This value could represent the percent of useful life remaining.

While these values do not have a time unit associated with them, they would allow for the compact form of a knot and spline model to be retained. The two alternatives may be preferable in a situation where rolling averages were used and replacement schedules were calculated based on amount of engine life used up



## Models:
### Linear regression with knots and splines.
Each feature that was included in the training data set was bootstrapped and plotted against cycles until failure. For example, the temperature recorded at location 50 at the low pressure turbine outlet (t50_lpt) looked like this:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/t50_lpt_bs_spline_analysis.png)

Each feature values were then evaluated to determine where the trend showed a marked increase or decrease in slope. At each location a knot value was chosen. These knot locations where then fit to the original data-frame which was transformed using each of the knots as a new feature. This procedure was duplicated for each predictive variable shown in this series of graphs:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/all_features_cycles_to_fail.png)

The model was trained using 15 of the 26 features provided:  
- Number of cycles the engine has been operating: time_cycles (integer)
- Temperature at the low pressure compressor outlet: t24_lpc (degrees Rankin = degrees Fahrenheit + 459.67)
- Temperature at the high pressure compressor outlet: t30_hpc (degrees Rankin)
- Temperature at the low pressure turbine outlet: t50_lpt (degrees Rankin)
- Pressure at the high pressure compressor: p30_hpc (psi)
- Rotation speed of the fan: nf_fan_speed (rpm)
- Rotation speed of the core: nc_core_speed (rpm)
- Pressure at the high pressure compressor: ps_30_sta_press (psi)
- Power factor (calculated): phi_fp_ps30
- Corrected fan speed: nrf_cor_fan_sp (psi)
- Corrected core speed: nrc_core_sp (rpm)
- bpr_bypass_rat: bypass ratio
- Ht Bleed enthalpy: htbleed_enthalpy (energy + (pressure * volume) )
- High pressure turbine coolant bleed: w31_hpt_cool_bl (lbm/s - pound mass per second)
- Low pressure turbine coolant bleed: w32_lpt_cool_bl (lbm/s - pound mass per second)



From these features knot locations were selected at specific values measured by the above sensors:
- time_cycles knot locations: 25, 50, 75, 120, 175 , 220, 240, 260, 280, 300
- t24_lpc knot locations: 641.5, 642,  642.5, 643.0 , 643.4, 644
- t30_hpc knot locations: 1584, 1588, 1593, 1598 , 1610
- t50_lpt knot locations: 1400, 1401, 1411, 1415, 1421, 1430, 1440
- p30_hpc knot locations: 552.2, 553.2, 554.8, 555, 555.5
- nf_fan_speed knot locations: 2388.1, 2388.15, 2388.2, 2388.3
- nc_core_speed knot locations: 9040, 9060, 9070, 9080, 9090
- ps_30_sta_press knot locations: 47, 47.2, 47.3, 47.45, 47.6, 47.8, 47.9
- phi_fp_ps30 knot locations: 520, 520.4 , 521.2, 522, 522.4, 523
- nrf_cor_fan_sp knot locations: 2388.6, 2388.2 , 2388.3, 2388.4
- nrc_core_sp knot locations: 8107.4 , 8117, 8127.5 , 8138.7 , 8149.4 , 8160 , 8171 , 8200 , 8250
- bpr_bypass_rat knot locations: 8.38, 8.41, 8.45, 8.49
- htbleed_enthalpy knot locations: 389, 390, 391, 392, 393, 394,395, 396, 397, 398, 399
- w31_hpt_cool_bl knot locations: 38.5, 38.7, 38.9, 39.1, 39.2
- w32_lpt_cool_bl knot locations: 23.14, 23.2, 23.32, 23.44



#### Pruning the knots:
After the model was trained, the variables were evaluated using a partial dependency plot.

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/partial_dependency_pipline.png)

Large variations at the edges were pruned and the model performance improved at the points where the predictions would move closer to 0.

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/pruned_partial_dependency_pipline.png)

Finally, additional knots needed to be pruned in nf_fan_speed and htbleed_enthalpy.

### Linear regression predicting number of cycles to fail.
The plots of the first model indicated that the features contained data that was increasing/decreasing at a rate that was accelerating as the engines approached the end of their life cycle. As shown in this graph:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/pred_vs_actual_reg_regression.png)


### Linear regression predicting the natural log of cycles to fail.
The curve in the predictions is a clear indication that a log transformation should be attempted. The first option was natural log transformation of the target value (cycles to failure). The log variables started at values as high as 5.8916 (362 life cycles remaining) and all engines continued down to a value of 0 (1 life cycle remaining). The results where significantly improved as shown in this plot:   

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/training_cycles_to_fail.png)

and from the test set:

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/test_cycles_to_fail.png)

Converting these estimates back into an estimate of cycles to fail can be accomplished by raising the value to the exponential of estimate:

![alt text](http://www.codecogs.com/gif.latex?e^{\hat{y}}=cycles)


### Training set vs Test set for each engine
The predictions and the actual values for the 80 engines in the training set are below.

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/all_80_failure_right_num_cycles.png)


The predictions for the 20 engines in the test set show patterns similar to the training data. This is an indication that, given the training data is representative of the real world data, the model will perform similarly when in production.

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/20_test_set_y_actual_num_cycles.png)


### The model output
The final output of the model was exported to a text file that included the coefficients, knot locations, and y intercept terms in a form that could be placed into a calculated field in tableau. The full text is available here [full text](https://github.com/fischtank44/flight_engineer/raw/master/outputs/tableau_format_formula.txt)

After placing the formula in Tableau the estimation of cycles remaining can calculated from the data as it is being read into the program. There is no need for any additional processing or for scripts to be run in order for model estimations to be completed. The entirety of this model can me transported in a 6kB text file.

[Tableau Public Site](https://public.tableau.com/profile/steven.fischbach#!/vizhome/FlightEngineer-v4_2/CautionsvsBigFormula)

![alt text](https://github.com/fischtank44/flight_engineer/raw/master/images/cycles_to_fail.png)
