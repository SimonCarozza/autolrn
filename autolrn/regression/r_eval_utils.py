from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel, RFECV
from autolrn.encoding import labelenc as lc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.metrics import r2_score, mean_squared_log_error, make_scorer
from time import time
from . import param_grids_distros as pgd
from . import neuralnets as nn
from .. import auto_utils as au
from autolrn.classification import eval_utils as eu
from random import randint
from scipy.stats import randint as sp_randint
from sklearn.svm import LinearSVR
from pandas import DataFrame
from sklearn.base import is_regressor
from sklearn.pipeline import Pipeline

EPSILON = np.finfo(float).eps 


def custom_rms_log_err(y_true, y_pred):
    """
    Root mean squared log error

    ---

    y_true: target date
    y_pred: predicted data
    """
    assert len(y_true) == len(y_pred)
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError(
                    "Root Mean Squared Logarithmic Error cannot be used when "
                    "targets contain negative values.")
    # msle = mean_squared_error(np.log1p(y_true), np.log1p(y_pred))
    msle = mean_squared_log_error(y_true, y_pred)
    return np.sqrt(msle)


# carefully prepare your target before using this...
def custom_rms_perc_err(y_true, y_pred):
    """
    Root mean squared percentage error

    ---

    y_true: target date
    y_pred: predicted data
    """
    assert len(y_true) == len(y_pred)
    if (y_true == 0).any():
        raise ZeroDivisionError(
                            "Root Mean Squared Percentage Error"
                            "cannot be used when targets contain zeros")

    # mspe = np.mean(np.square((y_true - y_pred)/np.clip(y_true, EPSILON, 1.)))
    mspe = np.mean(np.square((y_true - y_pred)/y_true))
    return np.sqrt(mspe)


def get_custom_scorer(scoring=None, y_train=None):
    # check scoring and y_train exist
    scorer=None
    if scoring == "neg_rmsle":
        if (y_train < 0).any():
            raise ValueError(
                    "Root Mean Squared Logarithmic Error cannot be used when "
                    "targets contain negative values.")
        else:
            scorer = make_scorer(
                custom_rms_log_err, greater_is_better=False
                )
    elif scoring == "neg_rms_perc_err":
        if (y_train == 0).any():
            raise ZeroDivisionError(
                    "Root Mean Squared Percentage Error cannot be used when "
                    "targets contain zeros.")
        else:
            scorer = make_scorer(
            custom_rms_perc_err, greater_is_better=False
            )
    else: 
        scorer = scoring

    return scorer


def split_and_encode_Xy(
        X, y, encoding='le', feat_scaler=True, tgt_scaler=True, 
        freqs=None, dummy_cols=10, ohe_dates=False,
        test_size=.25, feat_select=True, shuffle=True, enc_Xy=False, 
        X_test=None, scoring='r2'):
    """
    Splits X, y into train and test sub sets, encode them

    ---

    shuffle: set it to False to preserve items order
    """
    X_train, y_train, y_test = (None, None, None)
    # do not shuffle the data before splitting to respect row order
    if not enc_Xy:
        # check X, y are valid dataframes or numpy arrays...
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle)
    else:
        print()
        print("Encoding full data set 'X' -> 'X_train'")
        X_train = X
        y_train = y

    print("Let's have a look at the first row and output")
    print("X_train\n", X_train.head())
    print("y_train\n", y_train.head())
    print()

    if list(X.select_dtypes(include=["datetime"]).columns):
        print("datetime type found.")
        X_train = lc.get_date_features(X_train, freqs)
        if X_test is not None:
            X_test = lc.get_date_features(X_test, freqs)

        # print(X_train["Month"].head(3))

    if encoding == 'le':
        X_train = lc.dummy_encode(X_train.copy()).astype(np.float32)
        if X_test is not None:
            X_test = lc.dummy_encode(X_test.copy()).astype(np.float32)
    elif encoding == 'ohe':
        # do this for mixed label-onehot encoding !
        # X_train.reset_index(drop=True, inplace=True)
        # X_test.reset_index(drop=True, inplace=True)

        X_train = lc.get_dummies_or_label_encode(
          X_train.copy(), dummy_cols=dummy_cols, 
          ohe_dates=ohe_dates).astype(np.float32)
        # print("oheencoded X_train['month'] \n", X_train["Month"].head(3))

        if X_test is not None:
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

    if X_test is not None and X_test.isnull().values.any():
            X_test = X_test.fillna(X_test.median())

    print("After encoding, first row and output")
    print("X_train\n", X_train.head())
    print("X_train.columns\n", list(X_train.columns))
    print("y_train\n", y_train.head())
    print()

    scalers = (None, None)
    data_and_scalers = {"scalers": scalers}

    if feat_scaler:

        print("scaling train and test data")

        scaler = StandardScaler()
        # you're going to perform scaling at training time before finalization
        if not enc_Xy:
            X_train_scaled = scaler.fit_transform(X_train)
            X_train = DataFrame(
                data=X_train_scaled, columns=X_train.columns, index=X_train.index)
            
            print()
            print("X_train shape:", X_train.shape)
            if X_test is not None:
                X_test_scaled = scaler.transform(X_test)
                X_test = DataFrame(
                    data=X_test_scaled, columns=X_test.columns, index=X_test.index)
                print("X_test shape:", X_test.shape)
            
            print()
            print("After scaling...")
            print("X_train\n", X_train[:1])
            print("X_train type", type(X_train))
            if X_test is not None:
                print("X_test\n", X_test[:1])
                print("X_test type", type(X_test))
            print()

        scalers = (scaler, None)
        data_and_scalers["scalers"] = scalers

    print("scoring:", scoring)
    # tgt_scaler = False if scoring == 'neg_rmsle' else True
    # standard scaling introduces negative values, 
    # which can't be fed to log, hence to rmsle

    if tgt_scaler:
        print("Scaling target...")

        if scoring != 'neg_rmsle':
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
        else:
            y_scaler = MinMaxScaler()
            y_train = y_scaler.fit_transform(y_train.values.reshape(-1,1))

        print("y_train and its type\n", (y_train[:1], type(y_train)))

        if not enc_Xy:
            if scoring != 'neg_rmsle':
                y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()
            else:
                y_test = y_scaler.fit_transform(y_test.values.reshape(-1,1))

            print("y_test and its type\n", (y_test[:3], type(y_test)))

        scalers = (scalers[0], y_scaler)
        data_and_scalers["scalers"] = scalers

        print()

    # this works for classifiers
    # featsel_tuple = eu.create_feature_selector(X_train, None, seed)

    if feat_select and X_train.shape[1] > 10:

        lsvr = LinearSVR(max_iter=1e4)
        lsvr = lsvr.set_params(
            C=0.01, loss="squared_epsilon_insensitive", dual=False)
        # threshold=[1e-2, 1e-1] or in ["mean", "median"]
        thsd = "median"  # "median", "median"
        featselector = SelectFromModel(lsvr, threshold=thsd)
        # tscv_fs = TimeSeriesSplit(n_splits=5)
        # featselector = RFECV(lsvr, step=1, cv=tscv_fs)

        data_and_scalers["f_selector"] = featselector

        if not enc_Xy:
            # featselector = featsel_tuple[1]
            X_train_selected = featselector.fit_transform(X_train, y_train)
            xtr_indices = featselector.get_support()
            X_train = DataFrame(
                data=X_train_selected, 
                columns=X_train.columns[xtr_indices], 
                index=X_train.index)

            print("After feature selection...")
            print("X_train shape:", X_train.shape)
            if X_test is not None:
                X_test_selected = featselector.transform(X_test)
                xtt_indices = featselector.get_support()
                X_test = DataFrame(
                    data=X_test_selected, 
                    columns=X_test.columns[xtt_indices], 
                    index=X_test.index)

                print("X_test shape:",  X_test.shape)

    data_and_scalers["data"] = (X_train, X_test, y_train, y_test) 
    
    return data_and_scalers


def create_keras_regressors(input_dim, nb_epoch, batch_size, single_nn=None):

    keras_Reg_fcts = dict(
        baseline_nn_default_Reg=(nn.baseline_nn_model, {}),
        baseline_nn_smaller_Reg=(nn.baseline_nn_smaller_model, {}),
        larger_nn_Reg=(nn.larger_nn_model, {}),
        deep_nn_Reg=(nn.deep_nn_model, {}),
        deeper_nn_Reg=(nn.deeper_nn_model, {})
        )

    names_and_models = dict()

    if single_nn is None:
    
        if input_dim < 15:
            keras_Reg_fcts['larger_deep_nn_Reg'] = (nn.larger_deep_nn_model, {})
        
        for k, v in keras_Reg_fcts.items():
            names_and_models[k] = (KerasRegressor(
                build_fn=v[0], nb_epoch=nb_epoch,
                input_dim=input_dim, batch_size=batch_size,
                verbose=0), {})

    elif single_nn is not None and single_nn in keras_Reg_fcts:
        if single_nn=='larger_deep_nn_Reg' and input_dim >= 15:
            print(
                "input_dim = %d; not advisable to use %s" % single_nn)
            print("Switching to smaller model 'deeper_nn_Reg'")
            single_nn = 'deeper_nn_Reg'
        else:
            pass
        names_and_models[single_nn] = KerasRegressor(
            build_fn=keras_Reg_fcts[single_nn],  nb_epoch=nb_epoch,
            input_dim=input_dim, batch_size=batch_size,
            verbose=0), {}
    else:
        ValueError(
            "%s is not valid value for 'single_nn', valid values are "
            "[None, 'baseline_nn_default_Reg', 'baseline_nn_smaller_Reg'" 
            "'larger_nn_Reg', 'deep_nn_Reg', 'deeper_nn_Reg', "
            "'larger_deep_nn_Reg']")

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


def best_regressor_attributes(scoring=None):

    best_model_name = 'Worst'
    best_model = None
    best_reg_score = -np.inf
    best_reg_std = np.inf

    best_attributes = best_model_name, best_model, best_reg_score, best_reg_std

    return best_attributes


def set_features_params_for_model(
    X_train=None, time_dep=False, cv=3, model_name=None, model=None, 
    tuning='rscv', params=None, seed=0, nb_epoch=10, eval_phase=True):
    if time_dep:
        if not isinstance(cv, TimeSeriesSplit):
                raise TypeError(
                    "'%r' is not a valid type for time series splitting\n"
                    "valid type: 'sklearn.model_selection.TimeSeriesSplit'" %
                     type(cv))
        else:
            if not is_regressor(model):
                if not isinstance(model, KerasRegressor):
                    raise TypeError(
                        "non-sklearn regressor should be of type KerasRegressor")

    poly_features = None

    if model_name == 'Bagging_SVMReg':

        # testing phase, ensemble of SVRr w linear kernels already setup,
        # but you have to recreate param grid for hyperparameter opt.

        model.set_params(kernel='linear')

        n_estimators = 5
        bagging = BaggingRegressor(
            model, max_samples=1.0/n_estimators, n_estimators=n_estimators,
            oob_score=True, random_state=seed)

        model = bagging

        if eval_phase or tuning == 'rscv':
            params = {
                model_name + '__' +
                k: v for k, v in pgd.Bagging_param_grid.items()}
        
    elif model_name == "KerasReg" and not eval_phase:

        input_dim = int(X_train.shape[1])

        model, params = create_best_keras_reg_architecture(
            model_name, input_dim, model.get_params()['nb_epoch'], 
            pgd.Keras_param_grid)

    elif model_name in ('baseline_nn_default_Reg', 
        'baseline_nn_smaller_Reg', 'larger_nn_Reg', 'deep_nn_Reg', 
        'deeper_nn_Reg', 'larger_deep_nn_Reg') and not eval_phase:
    
        input_dim = int(X_train.shape[1])
        model, params = create_keras_regressors(
            input_dim, model.get_params()['nb_epoch'], 
            model.get_params()['batch_size'], model_name)

    elif model_name == "PolynomialRidgeReg":

        interact_only=False
        if X_train.shape[0] > 100000:
            interact_only=True
        poly_features = PolynomialFeatures(
            degree=5, interaction_only=interact_only)
        try:
            poly_features.fit(X_train)
        except MemoryError as me:
            print("MemoryError -- Unable to create polynomial features.")
            raise(me)
        except Exception as e:
            raise e
        else:
            print(
            "Successfully created PolynomialFeatures obj.:", poly_features)

    else:
        pass

    if not eval_phase:
        # # n_jobs=-2 to avoid burdening your PC
        if hasattr(model, 'n_jobs'):
            print("Using all available cores.")
            model.set_params(n_jobs=-1)

        if tuning is None and params is not None:
            print("Parameters are useless for testing default/optimized model.")
            params = None

    feats_and_params = (model, cv, tuning, params, poly_features)

    return feats_and_params


def save_regressor(
    estimator=None, name='', d_name=None, tuning='NoTuning', serial=0):
    if not serial:
        serial = "%04d" % randint(0, 1000)

    model_name = name

    if d_name is not None:
        name = d_name + "_" + name

    if tuning is None:
        tuning = 'NoTuning'
    elif tuning == 'rscv':
        pass
    elif tuning not in ('NoTuning', 'rscv'):
        raise ValueError(
                    "%s is not a valid value for variable 'tuning'"
                    "valid value are ['NoTuning', 'rscv']" % tuning)
    else:
        raise TypeError("'tuning' must be of type 'string'")

    f_name = name + '_' + tuning + '_' + serial

    # independent of model name
    # train fcts save pipelines, next v. save plain models
    if is_regressor(estimator):
            # can be a sklearn.pipeline(regressor)/regressor
            au.save_model(estimator, f_name + '.pkl')
    elif isinstance(estimator, Pipeline):
        if isinstance(estimator.named_steps[model_name], KerasRegressor):
            keras_f_name = au.create_keras_model_filename(f_name)
            estimator.named_steps[model_name].model.save(keras_f_name + '.h5')

            if(len(estimator.steps)) > 1:
                # estimator.named_steps[model_name].model = None
                # or
                # estimator[model_name].model = None
                estimator.steps.pop()
                f_name = name + '_' + tuning + '_for_keras_model_' + serial
                
                au.save_model(estimator, f_name + '.pkl')
        else:
            raise TypeError(
                    "% is not a sklearn pipeline containing a KerarRegressor"
                    "neither a KerasRegressor itself" % model_name)
    # here you could check whether you have plain KerasRegressor...
    else:
        raise TypeError(
            "% is neither a sklearn pipeline containing a valid Regressor"
            "nor a valid sklearn Regressor itself" % model_names)

    del estimator

    print()