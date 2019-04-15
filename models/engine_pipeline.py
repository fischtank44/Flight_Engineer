from sklearn.pipeline import Pipeline
from regression_tools.dftransformers import (
    ColumnSelector, 
    Identity,
    Intercept,
    FeatureUnion, 
    MapFeature,
    StandardScaler)
from basis_expansions.basis_expansions import (
    Polynomial, LinearSpline)

def fit_engine_pipeline(): 
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
        ('t30_hpc_spline', LinearSpline(knots=[ 1584, 1588, 1593, 1598 , 1610]))
    ])
    t50_fit = Pipeline([
        ('t50_lpt', ColumnSelector(name='t50_lpt')),
        ('t50_lpt_spline', LinearSpline(knots=[1400, 1401, 1411, 1415, 1421, 1430, 1440]))
    ])  
    p30_fit = Pipeline([
        ('p30_hpc', ColumnSelector(name='p30_hpc')),
        ('p30_hpc_spline', LinearSpline(knots=[ 552.2, 553.2, 554.8, 555, 555.5]))
    ])
    nf_fan_fit = Pipeline([
        ('nf_fan_speed', ColumnSelector(name='nf_fan_speed')),
        ('nf_fan_speed_spline', LinearSpline(knots=[2388.1, 2388.15, 2388.2, 2388.3]))
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
        ('nrf_cor_fan_sp_spline', LinearSpline(knots=[2388.6, 2388.2 , 2388.3, 2388.4]))
    ])
    nrc_core_fit = Pipeline([
        ('nrc_core_sp', ColumnSelector(name='nrc_core_sp')),
        ('nrc_core_sp_spline', LinearSpline(knots=[8107.4 , 8117, 8127.5 , 8138.7 , 8149.4 , 8160 , 8171 , 8200 , 8250]))
    ])
    bpr_bypass_fit = Pipeline([
        ('bpr_bypass_rat', ColumnSelector(name='bpr_bypass_rat')),
        ('bpr_bypass_rat_spline', LinearSpline(knots=[8.38, 8.41, 8.45, 8.49]))
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
    return feature_pipeline


