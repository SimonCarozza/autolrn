from sklearn.ensemble import BaggingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from time import time
import numpy as np
from . import param_grids_distros as pgd
from scipy.stats import randint as sp_randint
from sklearn.base import is_regressor


def train_test_process(
    best_model_name, estimators,
    X_train, X_test, y_train, y_test,
    y_scaler=None, tuning='rscv', cv=3, n_iter=10, nb_epoch=100,
    scoring='r2', random_state=0):

    if best_model_name not in ('Worst', 'DummyReg'):
        # test best model/estimator
        best_model = estimators[
            'SVMReg' if best_model_name=='Bagging_SVMReg' else
            best_model_name][0]
        params=None
        try:
            estimators[
                'SVMReg' if best_model_name=='Bagging_SVMReg' else
                best_model_name][1]
        except KeyError as ke:
            if best_model_name == "PolynomialRidgeReg":
                best_model = estimators["RidgeReg"][0]
                # print("tuning:", tuning)
                if tuning=='rscv':
                    params = estimators["RidgeReg"][1]
                    # print(best_model.get_params().keys())
            else:
                print(ke)
        except IndexError as ie:
            print(ie)
            print("No params here, this must be a test of optimized models.")
        except Exception as e:
            raise e
        else:
            if tuning == 'rscv':
                params = estimators[
                    'SVMReg' if best_model_name == 'Bagging_SVMReg' else
                    best_model_name][1]

        train_test_estimator(
            best_model_name, best_model, X_train, X_test, y_train, y_test,
            y_scaler=y_scaler, params=params, tuning=tuning, cv=cv,
            n_iter=n_iter, scoring=scoring, random_state=random_state)
    else:
        print("Unable to find a 'good-enough' regressor.")
        print("Current regressors suck!")


def train_test_estimator(
        best_model_name, best_model, X_train, X_test, y_train, y_test,
        y_scaler=None, params=None, tuning=None, cv=3, n_iter=10,
        nb_epoch=100, scoring=None, random_state=0, timeseries=False):
    """
    Tests optimized or plain estimator [needs refactorization]

    ----------------------------------
    """
    print()
    print("[task] === Model testing")
    print()

    if y_scaler is not None:
        y_train = y_scaler.inverse_transform(y_train)
        y_test = y_scaler.inverse_transform(y_test)

    if best_model_name != "Worst":
        if timeseries:
            if isinstance(cv, TimeSeriesSplit):
                if not is_regressor(best_model):
                    if not isinstance(best_model, KerasRegressor):
                        raise TypeError(
                            "non-sklearn regressor should be of type KerasRegressor")
            else:
                raise TypeError(
                    "'%r' is not a valid type for time series splitting\n"
                    "valid type is 'sklearn.model_selection.TimeSeriesSplit'" % cv)
        else:
            if best_model_name == 'Bagging_SVMReg':

                # testing phase, ensemble of SVRr w linear kernels already setup,
                # but you have to recreate param grid for hyperparameter opt.

                best_model.set_params(kernel='linear')

                n_estimators = 5
                bagging = BaggingRegressor(
                    best_model, max_samples=1.0/n_estimators, n_estimators=n_estimators,
                    oob_score=True, random_state=random_state)

                best_model = bagging

                if tuning == 'rscv':
                    params = {
                        best_model_name + '__' +
                        k: v for k, v in pgd.Bagging_param_grid.items()}

            elif best_model_name == "KerasReg":

                input_dim = int(X_train.shape[1])

                best_model, params = create_best_keras_reg_architecture(
                    best_model_name, input_dim, best_model.get_params()['nb_epoch'],
                    pgd.Keras_param_grid)

                if tuning == 'rscv':
                    for n in np.arange(0, 3):
                        params[best_model_name + '__units_' + str(n)] = sp_randint(
                            input_dim, 5*input_dim)

            elif best_model_name == "PolynomialRidgeReg":

                interact_only=False
                if X_train.shape[0] > 100000:
                    interact_only=True
                poly_features = PolynomialFeatures(
                    degree=5, interaction_only=interact_only)
                X_train = poly_features.fit_transform(X_train)
                X_test = poly_features.transform(X_test)

            else:
                pass

        if hasattr(best_model, 'n_jobs'):
            best_model.set_params(n_jobs=-2)

        if tuning is None and params is not None:
            print("Parameters are useless for testing default/optimized model.")
            params = None

        predicted = None

        if tuning is None and params is None:
            print("Fitting '%s' ..." % best_model_name)
            t0 = time()
            best_model.fit(X_train, y_train)
            t1 = time()
            predicted = best_model.predict(X_test)
        elif tuning == 'rscv' and params is not None:
            # check params is a dict w tuples (model, distro_params)
            print("Find best params of %s w RSCV and refit it..." % best_model_name)
            # for k, v in params.items():
            # 	print("\t", k)
            ppl = Pipeline([(best_model_name.lstrip("Polynomial"), best_model)])
            # print(ppl.get_params().keys())
            # input("press key...")
            try:
                estimator = RandomizedSearchCV(
                    ppl, param_distributions=params, cv=cv, iid=False,
                    n_iter=n_iter, n_jobs=-2, pre_dispatch="2*n_jobs", scoring=scoring,
                    refit=True, random_state=random_state, error_score=np.nan)
            except TypeError as te:
                print(te)
            except Exception as e:
                print(e)
            else:
                t0 = time()
                estimator.fit(X_train, y_train)
                t1 = time()

                # Using the score defined by scoring
                rscv_score = estimator.score(X_train, y_train)
                # mean_rscv_score = estimator.best_score_

                if scoring == 'neg_mean_squared_error':
                    rscv_score = -1.*rscv_score
                    # mean_rscv_score = -1.*mean_rscv_score

                print("Scoring [%s] of refitted estimator on train data: %1.3f"
                    % (scoring.strip('neg_'), rscv_score))
                # print("Mean cv score [%s] of the best_estimator: %1.3f"
                #     % (scoring.strip('neg_'), mean_rscv_score))

                predicted = estimator.predict(X_test)

                # print("best estimator: \n", estimator.best_estimator_)
                best_regressor = estimator.best_estimator_.named_steps[
                    best_model_name.lstrip("Polynomial")]

                best_rscv_params = estimator.best_params_
                if hasattr(best_regressor, 'oob_prediction_'):
                    best_oob_prediction = best_regressor.oob_prediction_
                    mse = mean_squared_error(y_train, best_oob_prediction)
                    rmse = np.sqrt(mean_squared_error(y_train, best_oob_prediction))
                    print("%s oob_predictions' MSE [%.2f] and RMSE [%.2f] on train data." % (
                        best_model_name, mse, rmse))
                    r2 = r2_score(y_test, predicted)
                    print("R2 oob_score on test data: %.2f" % (r2))
                if hasattr(best_regressor, 'oob_score_'):
                    best_oob_score = best_regressor.oob_score_
                    print("oob_score [%s] of refitted ensemble on train data: %1.3f" % (
                        scoring.strip('neg_'), best_oob_score))

                print()
                # as a check for fitting the same estimator
                # print("Best estimator has params:\n")
                # for param_name in sorted(best_rscv_params.keys()):
                #     print("\t%s: %r" % (param_name, best_rscv_params[param_name]))
                # print()

                best_model_params = {
                    k.split('__')[1]: v for k, v in best_rscv_params.items()
                }

                best_model.set_params(**best_model_params)
        elif tuning == 'rscv' and params is None:
            raise ValueError("rscv hyperparameter opt requires parameters dict.")
        else:
            raise ValueError("Check values of 'tuning' and 'params'")

        print()
        print("Best estimator '%s' has params:\n" % best_model_name)
        params = best_model.get_params()
        for param_name in sorted(params.keys()):
            print("\t%s: %r" % (param_name, params[param_name]))
        print()
        # y_test = np.array([int(yt) for yt in y_test])
        y_test = check_array(y_test, dtype='numeric', ensure_2d=False)
        print("y_test", y_test[0:3])
        print("predictions", predicted[0:3])
        # non-numerical y objects
        mse = mean_squared_error(y_test, predicted)
        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        print("%s predictions' MSE [%.2f] and RMSE [%.2f] on test data" % (
            best_model_name, mse, rmse))
        r2 = r2_score(y_test, predicted)
        print("R2 score on test data: %.2f" % (r2))
        evs = explained_variance_score(y_test, predicted)
        print("Explained Variance score on test data: %.2f." % (evs))
        print('Execution time for %s: %.2fs.' % (best_model_name, t1 - t0))

    print()
