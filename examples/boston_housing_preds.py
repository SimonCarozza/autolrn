from autolrn import auto_utils as au
from autolrn.regression import r_eval_utils as reu
import numpy as np
from pandas import read_csv, DataFrame
from autolrn.encoding import labelenc as lc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVR
from pkg_resources import resource_string
import os
from io import StringIO
from sklearn.externals import joblib as jl
from keras.models import load_model

# load test data

names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
    'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
url = "https://goo.gl/sXleFv"

# using data from the train df for education purpose
X_test = read_csv(url, delim_whitespace=True, names=names).iloc[-100:]
X_test.drop(["MEDV"], axis=1, inplace=True)

# statistical summary
description = X_test.describe()
print(description)
print()
print("Unseen X_test shape:", X_test.shape)
print("Unseen X_test types:\n", X_test.dtypes)


regressor, regressor_keras = None, None

try:
    regressor = jl.load("models/bostonhse_ExtraTreesReg_rscv_0576.pkl")
    regressor_keras = jl.load(
        "models/bostonhse_larger_deep_nn_Reg_NoTuning_for_keras_model_0124.pkl")
    regressor_keras.steps.append(
        ("bostonhse_larger_deep_nn_Reg_NoTuning_0124", 
         load_model("models/bostonhse_larger_deep_nn_Reg_NoTuning_0124.h5")))
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:
    print()

    try:
        predicted = regressor.predict(X_test)
    except ValueError as ve:
        print("ValueError:", ve)
    else:
        print("Regressor")
        print("Sample of first 15 predictions:")
        print(predicted[:15])
        print("Sample of last 15 predictions:")
        print(predicted[-15:])

    print()

    try:
        predicted = regressor_keras.predict(X_test)
    except ValueError as ve:
        print("ValueError:", ve)
    else:
        print("KerasRegressor")
        print("Sample of first 15 predictions:")
        print(predicted[:15])
        print("Sample of last 15 predictions:")
        print(predicted[-15:])

    print()

    # try:
    #     X_test = lc.get_date_features(X_test_orig)
    #     X_test = lc.dummy_encode(X_test.copy()).astype(np.float32)
        
    #     predicted = regressor_rmspe.predict(X_test)
    # except ValueError as ve:
    #     print("ValueError:", ve)
    # else:

    #     if X_test.isnull().values.any():
    #         X_test = X_test.fillna(X_test.median())

    #     print("Regressor for scoring='neg_rms_perc_err'")
    #     print("Sample of first 15 predictions:")
    #     print(predicted[:15])
    #     print("Sample of last 15 predictions:")
    #     print(predicted[-15:])
