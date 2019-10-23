from autolrn.regression import param_grids_distros as pgd
from autolrn import auto_utils as au
from autolrn.regression import r_eval_utils as reu
from autolrn.regression import evaluate as eu
from autolrn.regression import train as tr
from pandas import read_csv
import numpy as np
from pkg_resources import resource_string
import os
from io import StringIO
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyRegressor


if __name__ == "__main__":

    seed = 7
    np.random.seed(seed)

    d_name = "rossmann"

    # names = [
    #     "Store","DayOfWeek","Date","Sales","Customers","Open",
    #     "Promo","StateHoliday","SchoolHoliday"]

    # load data
    ross_bytes = resource_string(
        "autolrn", os.path.join("datasets", 'rossman_store_train.csv'))
    ross_file = StringIO(str(ross_bytes,'utf-8'))

    # n_rows = 15000  # 5000, 7000, 10000, 25000, 50000

    df = read_csv(
        ross_file, delimiter=",", 
        # names=names, 
        parse_dates=['Date'],
        # nrows = n_rows,
        dtype={"StateHoliday": "category"})

    print("Dataset shape:", df.shape)
    print("Dataset types:\n: ", df.dtypes)

    # statistical summary
    description = df.describe()
    print("Dataset description:\n", description)
    print()

    target = "Sales"

    df.dropna(subset=[target], inplace=True)

    print("Date column type", df["Date"].dtype)
    print()
    print("Open uniques:", df["Open"].unique())
    print()

    try:
        df["Date"] = df["Date"].astype('datetime64[D]')
    except Exception as e:
        print(e)
        df["Date"] = to_datetime(df["Date"], errors='coerce')
    # use this if you took a random sample of full df
    # df.sort_values(by=["Date"], inplace=True)
    df["DayOfWeek"] = df["DayOfWeek"].astype(str)
    df["Open"] = df["Open"].astype(str)
    df["Promo"] = df["Promo"].astype(str)
    df["StateHoliday"].replace(to_replace='0', value='n', inplace=True)
    df["SchoolHoliday"] = df["SchoolHoliday"].astype(str)
    df["Sales"] = df["Sales"].astype(str)  # .astype(int)

    print("After some processing")
    print("DayOfWeek uniques:", df["DayOfWeek"].unique())
    print("Open uniques:", df["Open"].unique())
    print("Promo uniques:", df["Promo"].unique())
    print("StateHoliday uniques:", df["StateHoliday"].unique())
    print("SchoolHoliday uniques:", df["SchoolHoliday"].unique())
    print()
    print(df[df["Open"] == '0'].head())
    print(len(df[df["Open"] == '0'].index))

    scoring = "r2" # 'neg_mean_squared_error', 'r2'

    # adding 'has_Sales' col
    if (df["Open"] == '0').any():
        # keeping zero sales records
        df['has_Sales'] = 1
        print("Some record has zero sale")
        # you don't have a target in unseen data
        # df['has_Sales'][df[target] == '0'] = 0
        df.loc[df["Open"] == '0', 'has_Sales'] = 0

    print()
    print("df shape after little cleaning: ", df.shape)

    X = df.drop([target], axis=1)
    y = df[target]

    print("X.dimensions: ", X.shape)
    print("y.dtypes:", y.dtypes)

    print()
    print("Let's have a look at the first row and output")
    print("X\n", X.head())
    print("y\n", y.head())

    ### 

    best_attr = reu.best_regressor_attributes()

    best_model_name, best_model, best_reg_score, best_reg_std = best_attr

    test_size = .1

    freqs = ['Year', 'Month', 'Day', 'Week']

    encoding = 'le'  # 'le', 'ohe'

    splitted_data = reu.split_and_encode_Xy(
        X, y, 
        encoding=encoding, freqs=freqs, 
        # dummy_cols=7, # 31, 52 
        test_size=test_size, 
        shuffle=False, scoring=scoring)

    X_train, X_test, y_train, y_test = splitted_data["data"]
    scaler, tgt_scaler = splitted_data["scalers"]
    featselector = None
    if 'f_selector' in splitted_data:
        featselector = splitted_data["f_selector"]

    print()

    tscv = TimeSeriesSplit(n_splits=2)
    # cv = tscv

    estimators = dict(pgd.full_search_models_and_parameters)

    # adding keras regressor to candidate estimators
    input_dim = int(X_train.shape[1])
    nb_epoch = au.select_nr_of_iterations('nn')

    refit, nested_cv, tuning = eu.select_cv_process(cv_proc='cv')

    print(
        "refit:'%s', nested_cv:'%s', tuning:'%s'" % (refit, nested_cv, tuning))

    if refit:

        # This is gonna work even with no prior optimization

        keras_reg_name = "KerasReg"
        keras_nn_model, keras_param_grid = reu.create_best_keras_reg_architecture(
            keras_reg_name, input_dim, nb_epoch, pgd.Keras_param_grid)

        estimators[keras_reg_name] = (keras_nn_model, keras_param_grid)

    else:

        keras_regressors = reu.create_keras_regressors(input_dim, nb_epoch, batch_size=32)
        estimators.update(keras_regressors)

    print()

    print("[task] === Model evaluation")
    print("*** Best model %s has score %.3f +/- %.3f" % (
        best_model_name, best_reg_score, best_reg_std))
    print()

    best_attr = eu.evaluate_regressor(
        'DummyReg', DummyRegressor(strategy="median"), 
        X_train, y_train, tscv, scoring, best_attr, time_dep=True)

    # cv_proc in ['cv', 'non_nested', 'nested']
    refit, nested_cv, tuning = eu.select_cv_process(cv_proc='cv')

    best_model_name, _, _ , _ = eu.get_best_regressor_attributes(
        X_train, y_train, estimators, best_attr, scoring, 
        refit=refit, nested_cv=nested_cv,
        cv=tscv, time_dep=True, random_state=seed)

    time_dep=True  
    print("set time_dep=%s" % time_dep)

	# train phase, no saving: scaler and featselector are useless
    tested = tr.train_test_process(
        best_model_name, estimators, X_train, X_test, y_train, y_test,
        scaler=scaler, y_scaler=tgt_scaler, feat_selector=featselector, 
        tuning=tuning, cv=tscv, scoring=scoring, random_state=seed, 
        time_dep=time_dep)

    # if a best model has been successfully tested, proceed to full training
    # for prediction on unseen data
    if tested:
        print()
        print("[task] === Train %s for predictions on unseen data" % best_model_name)

        X_train = X
        y_train = y

        del X, y

        encoded_data = reu.split_and_encode_Xy(
        X_train, y_train, encoding='le', feat_select=True, enc_Xy=True, 
        scoring=scoring)

        X_train, _, y_train, _ = encoded_data["data"] 
        scaler, tgt_scaler = encoded_data["scalers"]
        featselector = None
        if 'f_selector' in splitted_data:
            featselector = splitted_data["f_selector"]

        tr.train_test_process(
            best_model_name, estimators, X_train, X_test, y_train, y_test,
            scaler=scaler, y_scaler=tgt_scaler, feat_selector=featselector, 
            tuning=tuning, cv=tscv, scoring=scoring, random_state=seed, 
            time_dep=time_dep, test_phase=False, d_name=d_name)

    print()
    print("End of program\n")
