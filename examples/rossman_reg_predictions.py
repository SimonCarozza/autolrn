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

# load test data

ross_bytes = resource_string(
"autolrn", os.path.join("datasets", 'rossman_store_test.csv'))
ross_test = StringIO(str(ross_bytes,'utf-8'))

X_test = read_csv(
    ross_test, delimiter=",", 
    parse_dates=['Date'],
    dtype={"StateHoliday": "category"})

try:
    X_test["Date"] = X_test["Date"].astype('datetime64[D]')
except Exception as e:  
    print(e)
    X_test["Date"] = to_datetime(X_test["Date"], errors='coerce')
X_test["DayOfWeek"] = X_test["DayOfWeek"].astype(str)
X_test["Open"] = X_test["Open"].astype(str)
X_test["Promo"] = X_test["Promo"].astype(str)
X_test["StateHoliday"].replace(to_replace='0', value='n', inplace=True)
X_test["SchoolHoliday"] = X_test["SchoolHoliday"].astype(str)

print("After some processing")
print("DayOfWeek uniques:", X_test["DayOfWeek"].unique())
print("Open uniques:", X_test["Open"].unique())
print("Promo uniques:", X_test["Promo"].unique())
print("StateHoliday uniques:", X_test["StateHoliday"].unique())
print("SchoolHoliday uniques:", X_test["SchoolHoliday"].unique())
print()
# df.drop(["Store"], axis=1, inplace=True)

X_test['has_Sales'] = 1
print("Some record has zero sale")
# you don't have a target in unseen data
# df['has_Sales'][df[target] == '0'] = 0
X_test.loc[X_test["Open"] == '0', 'has_Sales'] = 0

# statistical summary
description = X_test.describe()
print(description)
print()
print("Unseen X_test shape:", X_test.shape)
print("Unseen X_test types:\n", X_test.dtypes)

# you need a pickled pipeline, this is just a workaround

X_test = lc.get_date_features(X_test)
X_test = lc.dummy_encode(X_test.copy()).astype(np.float32)

if X_test is not None and X_test.isnull().values.any():
    X_test = X_test.fillna(X_test.median())

print()

regressor = None

try:
    # metric: 'r2'
    # _0630 : 'Store' col dropped out
    reg_r2 = jl.load("models/rossmann_GBoostingReg_NoTuning_0843.pkl")
except OSError as oe:
    print(oe)
except Exception as e:
    raise e
else:
    print()

    try:
        predicted = reg_r2.predict(X_test)
    except ValueError as ve:
        print("ValueError:", ve)
    else:
        print("GBoostinReg optimized for 'r2'")
        print("Sample of first 15 predictions:")
        print(predicted[:15])
        print("Sample of last 15 predictions:")
        print(predicted[-15:])

