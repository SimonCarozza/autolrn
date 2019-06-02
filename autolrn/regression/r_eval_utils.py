from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel, RFECV
from autolrn.encoding import labelenc as lc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from time import time
from . import param_grids_distros as pgd
from . import neuralnets as nn
from .. import auto_utils as au
from autolrn.classification import eval_utils as eu
from scipy.stats import randint as sp_randint
from sklearn.svm import LinearSVR


def split_and_encode_Xy(
        X, y, encoding='le', freqs=None, dummy_cols=10, ohe_dates=False,
        test_size=.25, f_sel=True, shuffle=True):
    """
    Splits X, y into train and test sub sets, encode them

    ---

    shuffle: set it to False to preserve items order
    """
    # do not shuffle the data before splitting to respect row order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle)

    print("Let's have a look at the first row and output")
    print("X_train\n", X_train.head())
    print("y_train\n", y_train.head())
    print()

    if list(X.select_dtypes(include=["datetime"]).columns):
        print("datetime type found.")
        X_train = lc.get_date_features(X_train, freqs)
        X_test = lc.get_date_features(X_test, freqs)

        # print(X_train["Month"].head(3))

    if encoding == 'le':
        X_train = lc.dummy_encode(X_train.copy()).astype(np.float32)
        X_test = lc.dummy_encode(X_test.copy()).astype(np.float32)
    elif encoding == 'ohe':
        # do this for mixed label-onehot encoding !
        # X_train.reset_index(drop=True, inplace=True)
        # X_test.reset_index(drop=True, inplace=True)

        X_train = lc.get_dummies_or_label_encode(
          X_train.copy(), dummy_cols=dummy_cols, 
          ohe_dates=ohe_dates).astype(np.float32)
        # print("oheencoded X_train['month'] \n", X_train["Month"].head(3))

        X_test = lc.get_dummies_or_label_encode(
          X_test.copy(), dummy_cols=dummy_cols, 
          ohe_dates=ohe_dates).astype(np.float32)

        X_test = eu.reorder_ohencoded_X_test_columns(X_train, X_test)
    else:
        raise ValueError(
            "%r is not a valid value for var 'encoding', \n"
            "valid values are in ['le', 'ohe']" % encoding)

    print()

    if X_train.isnull().values.any():
        X_train = X_train.fillna(X_train.median())

    if X_test.isnull().values.any():
        X_test = X_test.fillna(X_test.median())

    print("After encoding, first row and output")
    print("X_train\n", X_train.head())
    print("X_train.columns\n", list(X_train.columns))
    print("y_train\n", y_train.head())
    print()

    data_and_scalers = {}

    print("scaling train and test data")

    scaler = StandardScaler()
    # scaler.fit(X_train, y_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print()
    print("Shapes of train and test:", X_train.shape, X_test.shape)
    print()

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
    y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()

    # y_train, y_test = y_train.values, y_test.values

    data_and_scalers["scalers"] = (scaler, y_scaler)

    print("After scaling...")
    print("X_train\n", X_train[:1])
    print("X_test\n", X_test[:1])
    print("X type (train, test)", (type(X_train), type(X_test)))
    print()
    print("y_train\n", y_train[:1])
    print("y_test\n", y_test[:1])
    print("y type (train, test)", (type(y_train), type(y_test)))
    print()
    # input("Press key to continue...")
    # print()

    # this works for classifiers
    # featsel_tuple = eu.create_feature_selector(X_train, None, seed)

    if f_sel:
        if X_train.shape[1] > 10 or X_test.shape[1] > 10:

            lsvr = LinearSVR(max_iter=1e4)
            lsvr = lsvr.set_params(
                C=0.01, loss="squared_epsilon_insensitive", dual=False)
            # threshold=[1e-2, 1e-1] or in ["mean", "median"]
            thsd = "median"  # "median", "median"
            featselector = SelectFromModel(lsvr, threshold=thsd)
            # tscv_fs = TimeSeriesSplit(n_splits=5)
            # featselector = RFECV(lsvr, step=1, cv=tscv_fs)

            data_and_scalers["f_selector"] = featselector

            # featselector = featsel_tuple[1]
            X_train = featselector.fit_transform(X_train, y_train)
            X_test = featselector.transform(X_test)

            print("After feature selection...")
            print("Shapes of train and test:", X_train.shape, X_test.shape)

    data_and_scalers["data"] = (X_train, X_test, y_train, y_test) 
    
    return data_and_scalers


def create_keras_regressors(input_dim, nb_epoch, batch_size):

    keras_Reg_fcts = dict(
        baseline_nn_default_Reg=(nn.baseline_nn_model, {}),
        baseline_nn_smaller_Reg=(nn.baseline_nn_smaller_model, {}),
        larger_nn_Reg=(nn.larger_nn_model, {}),
        deep_nn_Reg=(nn.deep_nn_model, {}),
        deeper_nn_Reg=(nn.deeper_nn_model, {})
        )

    if input_dim < 15:
        keras_Reg_fcts['larger_deep_nn_Reg'] = (nn.larger_deep_nn_model, {})

    names_and_models = dict()
    
    for k, v in keras_Reg_fcts.items():
        names_and_models[k] = (KerasRegressor(
            build_fn=v[0], nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size,
            verbose=0), {})
            
    return names_and_models


def create_best_keras_reg_architecture(
        keras_reg_name, input_dim, nb_epoch, keras_param_grid):
    """
    Tune KerasReg's hyperparameters using Randomized Search CV

    ------
    """

    for n in np.arange(0, 3):
        keras_param_grid[keras_reg_name + '__units_' + str(n)] = sp_randint(
            input_dim, 5*input_dim)

    keras_nn_model = KerasRegressor(
        build_fn=nn.tunable_deep_nn, nb_epoch=nb_epoch,
        input_dim=input_dim, verbose=0)

    return keras_nn_model, keras_param_grid


def best_regressor_attributes():

    best_model_name = 'Worst'
    best_model = None
    best_reg_score = -np.inf
    best_reg_std = np.inf

    best_attributes = best_model_name, best_model, best_reg_score, best_reg_std

    return best_attributes

