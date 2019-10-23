from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyRegressor
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from autolrn.regression import param_grids_distros as pgd
from autolrn.regression import r_eval_utils as reu
from autolrn.regression import evaluate as eu
from autolrn.regression import train as tr
from autolrn import auto_utils as au
from pandas import read_csv
import numpy as np
from pkg_resources import resource_string
import os
from io import StringIO


if __name__ == "__main__":

    seed = 7
    np.random.seed(seed)

    names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
        'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    url = "https://goo.gl/sXleFv"
    df = read_csv(url, delim_whitespace=True, names=names)

    d_name = "bostonhse"

    # load data
    
    print(df.shape)
    print("Dataset types: ", df.dtypes)

    # statistical summary
    description = df.describe()
    print(description)
    print()

    target = "MEDV"

    df.dropna(subset=[target], inplace=True)

    X = df.drop([target], axis=1)
    # X = X[0::100]
    y = df[target]
    # Y = Y[0::100]

    print("X.dimensions: ", X.shape)
    print("y.dtypes:", y.dtypes)

    print()
    print("Let's have a look at the first row and output")
    print("X\n", X.head())
    print("X\n", y.head())
    print()

    best_model_name = 'Worst'
    best_model = None
    best_reg_score = -np.inf
    best_reg_std = np.inf

    best_attr = best_model_name, best_model, best_reg_score, best_reg_std

    test_size = .15

    splitted_data = reu.split_and_encode_Xy(
        X, y, encoding='le', test_size=test_size, feat_select=True)

    X_train, X_test, y_train, y_test = splitted_data["data"]
    scaler, tgt_scaler = splitted_data["scalers"]
    featselector = None
    if 'f_selector' in splitted_data:
        featselector = splitted_data["f_selector"]

    print()

    # kfold =  KFold(n_splits=5, shuffle=True, random_state=seed)
    # cv = kfold

    scoring = "neg_mean_squared_error" # 'neg_mean_squared_error', 'r2'

    estimators = dict(pgd.full_search_models_and_parameters)

    # cv_proc in ['cv', 'non_nested', 'nested']
    refit, nested_cv, tuning = eu.select_cv_process(cv_proc='nested')

    print(
        "refit:'%s', nested_cv:'%s', tuning:'%s'" % (refit, nested_cv, tuning))

    n_iters = 15
    input_dim = int(X_train.shape[1])
    nb_epoch = au.select_nr_of_iterations('nn')

    if refit:

        # This is gonna work even with no prior optimization

        keras_reg_name = "KerasReg"
        keras_nn_model, keras_param_grid = reu.create_best_keras_reg_architecture(
            keras_reg_name, input_dim, nb_epoch, pgd.Keras_param_grid)

        # uncomment to see KerasRegressor's parameters
        # print()
        # print("KerasReg:\n", keras_nn_model)
        # print("Keras Params:\n", keras_param_grid)
        # print()
        estimators[keras_reg_name] = (keras_nn_model, keras_param_grid)

    else:

        keras_regressors = reu.create_keras_regressors(input_dim, nb_epoch, batch_size=32)
        estimators.update(keras_regressors)

    print()

    print("[task] === Model evaluation")
    print("*** Best model %s has score %.3f +/- %.3f" % (
        best_model_name, best_reg_score, best_reg_std))
    print()

    # cv=3
    best_attr = eu.evaluate_regressor(
        'DummyReg', DummyRegressor(strategy="median"), 
        X_train, y_train, 3, scoring, best_attr)

    best_reg_attributes = eu.get_best_regressor_attributes(
        X_train, y_train, estimators, best_attr, scoring, 
        refit=refit, nested_cv=nested_cv, random_state=seed)

    best_model_name = best_reg_attributes[0]
    # best_model = best_reg_attributes[1]   
    # best_reg_score = best_reg_attributes[2]
    # best_reg_std = best_reg_attributes[3]

    print()
    print()

    # training-testing process
    if best_model_name not in ('Worst', 'DummyReg'):
        # test best model/estimator

        best_model = estimators[
            'SVMReg' if best_model_name=='Bagging_SVMReg' else 
            best_model_name][0]
        if tuning=='rscv':
                params = estimators[
                'SVMReg' if best_model_name=='Bagging_SVMReg' else 
                best_model_name][1]

        if best_model_name == "PolynomialRidgeReg":
                parameters = estimators["RidgeReg"][1]
        else:
                parameters = estimators[best_model_name][1]

        tested, _ = tr.train_test_estimator(
            best_model_name, best_model, 
            X_train, X_test, y_train, y_test, y_scaler=tgt_scaler,
            params=parameters, tuning=tuning, n_iter=n_iters, 
            scoring=scoring, random_state=seed)

        # if a best model has been successfully tested, proceed to full training
        # for prediction on unseen data
        if tested:
            print()
            print("[task] === Train %s for predictions on unseen data" 
                  % best_model_name)

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

            # this is going to save a pipeline including pre-processors
            tr.train_test_estimator(
                best_model_name, best_model, 
                X_train, _, y_train, _, scaler=scaler, y_scaler=tgt_scaler,
                feat_selector=featselector, params=parameters, tuning=tuning, 
                scoring=scoring, random_state=seed, test_phase=False, 
                d_name=d_name)
    else:
        print("Unable to find a 'good-enough' regressor.")
        print("Current regressors suck!")

    print()
    print("End of program\n")