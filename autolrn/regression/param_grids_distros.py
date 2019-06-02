"""Hyperparameter grids and distros for GridSearchCV and RandomizedSearchCV."""

# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge  # alpha
# from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge, ARDRegression

# PolynomialFeatures ...

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR, LinearSVR

from scipy.stats import expon as sp_exp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_unif
from scipy.stats import beta as sp_beta
import numpy as np

xgb_import = 0
try:
    from xgboost import XGBRegressor
except ImportError as ie:
    print(ie)
else:
    xgb_import = 1

import warnings
warnings.filterwarnings("ignore")

seed = 7

# list of candidate hyperparameter grids for GSCV and distros for RSCV
# param spaces for searching with Hyperopt

# LinearRegression

# LR_gscv_param_grid = dict(
#     LReg__C=[0.01, 0.1, 1., 10., 100.],
#     )

# LR_param_grid = dict(
#     LReg__C=sp_exp(scale=100),
#     )

Rdg_gscv_param_grid = dict(
    RidgeReg__alpha=[.1, 0.5, 1., 5., 10.]
)

Rdg_param_distro = dict(
    RidgeReg__alpha=sp_exp(scale=10)
)

# BayesianRidge , ARD Bayes

BYR_gscv_param_grid = dict(
    BayesRidgeReg__tol=[0.0001, 0.001, 0.01, 0.1, 1],
    # BayesRidgeReg__alpha1=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    # BayesRidgeReg__alpha2=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
)

BYR_param_distro = dict(
    BayesRidgeReg__tol=sp_exp(scale=1)
)

ARD_param_distro = dict(
    ARDBayesReg__tol=BYR_param_distro["BayesRidgeReg__tol"]
)

# ElasticNet

EN_gscv_param_grid = dict(
    ElasticNetReg__l1_ratio=[.1, .5, .7, .9, .95, .99, 1]
)

EN_param_distro = dict(
    ElasticNetReg__l1_ratio=sp_exp(scale=1)
)


# KNeighborsReg

KNR_gscv_param_grid = dict(
    KNeighborsReg__weights=['uniform', 'distance'],
    KNeighborsReg__n_neighbors=[5, 10, 30, 50]
    )

KNR_param_distro = dict(
    KNeighborsReg__weights=['uniform', 'distance'],
    KNeighborsReg__n_neighbors=sp_randint(5, 50)
    )

# RandomForestReg

RFR_gscv_param_grid = dict(
    RandomForestReg__n_estimators=[10, 30, 50, 100, 200, 500, 1000],
    RandomForestReg__criterion=['mse', 'mae'],
    RandomForestReg__max_features=[None, .75, .5, 'sqrt'],
    # RandomForestReg__max_features=[0.25, 'auto', 'sqrt', 'log2'],
    RandomForestReg__min_samples_leaf=[1, 3, 5],
    RandomForestReg__max_depth=[3, 5, 7, 10]
    )

RFR_param_distro = RFR_gscv_param_grid
RFR_param_distro['RandomForestReg__n_estimators'] = sp_randint(10, 500)  # 100, 1000
RFR_param_distro['RandomForestReg__max_depth'] = sp_randint(3, 10)
RFR_param_distro['RandomForestReg__min_samples_leaf'] = sp_randint(1, 5)

# ExtraTrees Reg

ETR_param_distro = dict(
    ExtraTreesReg__n_estimators=RFR_param_distro[
        'RandomForestReg__n_estimators'],
    # ExtraTreesClf_2nd__n_estimators=sp_randint(100, 1000),
    ExtraTreesReg__criterion=RFR_param_distro['RandomForestReg__criterion'],
    ExtraTreesReg__max_features=RFR_param_distro[
        'RandomForestReg__max_features'],
    ExtraTreesReg__min_samples_leaf=RFR_param_distro[
        'RandomForestReg__min_samples_leaf'],
    ExtraTreesReg__max_depth=sp_randint(3, 7)
)


# GradientBoosting Reg

GBR_gscv_param_grid = dict(
    GBoostingReg__learning_rate=[.0001, .001, .01, .1, 10.],
    GBoostingReg__n_estimators=[100, 200, 500, 1000],
    GBoostingReg__subsample=[.3, .5, .8, 1.],
    GBoostingReg__max_features=[None, .75, .5, 'log2'],
    # GBoostingReg__max_features=[0.25, 'auto', 'sqrt', 'log2'],
    GBoostingReg__min_samples_leaf=[1, 3, 5],
    GBoostingReg__max_depth=[3, 5, 7, 10],
    )

second_half = sp_unif(0.5, 0.5)

GBR_param_distro = dict(
    # GBoostingReg__loss=['deviance', 'exponential'],
    GBoostingReg__learning_rate=sp_beta(3, 1),  # sp_exp(10),
    GBoostingReg__n_estimators=sp_randint(10, 500),  # 100, 1000
    GBoostingReg__subsample=second_half,
    GBoostingReg__max_features=GBR_gscv_param_grid[
        'GBoostingReg__max_features'],
    GBoostingReg__min_samples_leaf = RFR_param_distro[
        'RandomForestReg__min_samples_leaf'],
    GBoostingReg__max_depth=sp_randint(3, 10),
    )

# SVR

SVR_gscv_param_grid = dict(
    SVMReg__C=[0.01, .1, 1, 10, 100],
    SVMReg__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
    SVMReg__degree=np.arange(1, 5),
    SVMReg__gamma=['auto', 0.001, 0.01, 0.1, 1., 10., 100.],
    SVMReg__coef0=[0., 1., 5., 10., 20., 50.]
    )

SVR_param_distro = dict(
    SVMReg__C=sp_exp(100),
    SVMReg__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
    SVMReg__degree=sp_randint(1, 5),
    SVMReg__gamma=sp_exp(100),    # ['auto', sp_exp(100)]
    SVMReg__coef0=sp_exp(scale=50))


# LinearSVR

LinSVR_gscv_param_grid = dict(
    LinearSVMReg__C=[0.01, 0.1, 1, 5, 10, 100],
    # LinearSVMReg__dual=[False, True],
    LinearSVMReg__tol=[0.0001, 0.001, 0.01, 0.1, 1])

LinSVR_param_distro = dict(
    LinearSVMReg__C=sp_exp(scale=100),
    LinearSVMReg__tol=sp_exp(scale=1),
    # LinearSVMReg__dual=[False, True],
    )


one_to_left = sp_beta(3, 1)   # sp_beta(10, 1)
from_zero_positive = sp_exp(0, 50)
# second_half = sp_unif(0.5, 1-0.5)


# XGBoost

if xgb_import:
    XGBR_param_distro = dict(
        XGBReg__n_estimators=sp_randint(10, 500),  # 100, 1000
        XGBReg__max_depth=sp_randint(3, 40),
        XGBReg__learning_rate=one_to_left,
        XGBReg__gamma=sp_beta(0.5, 1),
        XGBReg__reg_alpha=from_zero_positive,
        XGBReg__min_child_weight=from_zero_positive,
        XGBReg__subsample=second_half,
        )

Bagging_gscv_param_grid = dict(
    n_estimators=[3, 5, 10, 15, 30], max_samples=[0.1, 0.2, 0.3, 0.5, 1.0])

Bagging_param_grid = dict(
    n_estimators=[3, 5, 10, 15, 30], max_samples=sp_unif(scale=1))

# KerasRegressor

# define nr of units at run time
Keras_param_grid = dict(
    KerasReg__batch_size=[8, 16, 32, 64, 128],  # sp_randint(8, 128),
    # KerasReg__n_layer=[0, 1, 2, 3] # sp_randint(0, 3), 
    # KerasReg__power=sp_randint(1, 3) # sp_exp(scale=3)
    )

for n in np.arange(0, 3):
    Keras_param_grid["KerasReg__dropout_rate_" + str(n)]=sp_unif(scale=.9)

###

# 'DummyReg' initialized at run-time, 'ARDBayesReg' slow af.

full_search_models_and_parameters = {
    'RidgeReg': (Ridge(), Rdg_param_distro),
    # 'ARDBayesReg': (ARDRegression(), ARD_param_distro),
    'ElasticNetReg': (ElasticNet(), EN_param_distro),
    'KNeighborsReg': (
        KNeighborsRegressor(n_neighbors=10), KNR_param_distro),
    'RandomForestReg': (
        RandomForestRegressor(
            random_state=seed, n_estimators=100, 
            max_features=.75,
            min_samples_leaf=5, 
            max_depth=5, 
            oob_score=True), 
        RFR_param_distro),
    'ExtraTreesReg': (
        ExtraTreesRegressor(
            bootstrap=True, n_estimators=100, 
            # max_features=.75,
            min_samples_leaf=5, 
            max_depth=5, 
            random_state=seed,
            oob_score=True),
            ETR_param_distro),
    'GBoostingReg': (
        GradientBoostingRegressor(
            loss='huber', criterion='friedman_mse', 
            max_features=.75,
            subsample=.3, 
            min_samples_leaf=5, 
            max_depth=5, 
            random_state=seed), 
            GBR_param_distro),
    'LinearSVMReg': (LinearSVR(max_iter=1e4), LinSVR_param_distro),
    # 'SVMReg': (SVR(kernel='rbf', gamma='scale'), SVR_param_distro)
}

if xgb_import:
    full_search_models_and_parameters['XGBReg'] =\
    (XGBRegressor(
        subsample=.3, 
        max_depth=5, 
        min_child_weight=.75,
        random_state=seed), 
    XGBR_param_distro)


# Lightly pre-optimized models to be used 
# as starting point for classification problems...
# let's "transfer" that to regression -- inexact classification
# R. Olson et al.: https://arxiv.org/abs/1708.05070
starting_point_models_and_params = {
    'RandomForestReg': RandomForestRegressor(
        n_estimators=500, max_features=.25, criterion='mse',
        oob_score=True, random_state=seed),
    'ExtraTreesReg': ExtraTreesRegressor(
        n_estimators=1000, max_features='log2', criterion='mse',
        random_state=seed),
    'GBoostingReg': GradientBoostingRegressor(
        loss='huber', n_estimators=500,  max_features='log2',
        random_state=seed),
    'SVMReg': SVR(
        C=.01, gamma=.1, kernel='poly', degree=3, coef0=10.),
    }
