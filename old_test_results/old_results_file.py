
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


#### Begining of the linear spline transformation parameters    #######
# linear_spline_transformer = LinearSpline(knots=[10, 35, 50, 80, 130, 150, 200, 250, 300])

# linear_spline_transformer.transform(df1['cycles_to_fail']).head()

# cement_selector = ColumnSelector(name='cycles_to_fail')
# cement_column = cement_selector.transform('cycles_to_fail')
# linear_spline_transformer.transform(cement_column).head()






