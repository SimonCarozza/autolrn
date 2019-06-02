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
        X, y, encoding='le', test_size=test_size, f_sel=True)

    X_train, X_test, y_train, y_test = splitted_data["data"]
    tgt_scaler = splitted_data["scalers"][1]

    print()

    # kfold =  KFold(n_splits=3, shuffle=True, random_state=seed)
    # cv = kfold

    scoring = "r2" # 'neg_mean_squared_error'

    estimators = dict(pgd.full_search_models_and_parameters)

    # This is gonna work even with no prior optimization

    input_dim = int(X_train.shape[1])
    nb_epoch = au.select_nr_of_iterations('nn')

    keras_reg_name = "KerasReg"

    keras_nn_model, keras_param_grid = reu.create_best_keras_reg_architecture(
        keras_reg_name, input_dim, nb_epoch, pgd.Keras_param_grid)

    # print()
    # print("KerasReg:\n", keras_nn_model)
    # print("Keras Params:\n", keras_param_grid)
    print()

    estimators[keras_reg_name] = (keras_nn_model, keras_param_grid)

    print("[task] === Model evaluation")
    print("*** Best model %s has score %.3f +/- %.3f" % (
        best_model_name, best_reg_score, best_reg_std))
    print()

    # cv=3
    best_attr = eu.evaluate_regressor(
        'DummyReg', DummyRegressor(strategy="median"), 
        X_train, y_train, 3, scoring, best_attr)

    # cv_proc in ['cv', 'non_nested', 'nested']
    refit, nested_cv, tuning = eu.select_cv_process(cv_proc='cv')

    print(
        "refit:'%s', nested_cv:'%s', tuning:'%s'" % (refit, nested_cv, tuning))

    best_reg_attributes = eu.get_best_regressor_attributes(
        X_train, y_train, estimators, best_attr, scoring, 
        refit=refit, nested_cv=nested_cv, random_state=seed)

    best_model_name = best_reg_attributes[0]

    print()

    # tuning == 'rscv' by default
    tr.train_test_process(
        best_model_name, estimators, X_train, X_test, y_train, y_test,
        y_scaler=tgt_scaler, tuning=tuning, scoring='r2', random_state=seed)

    print()
    print("End of program\n")