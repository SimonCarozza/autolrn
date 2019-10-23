from sklearn.ensemble import BaggingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
from sklearn.utils import check_array
from sklearn import metrics as me
from time import time
import numpy as np
from . import param_grids_distros as pgd
from . import r_eval_utils as reu
from scipy.stats import randint as sp_randint
from sklearn.base import is_regressor


def train_test_process(
    best_model_name, estimators, X_train, X_test, y_train, y_test, 
    scaler=None, y_scaler=None, feat_selector=None, tuning='rscv', cv=3, 
    n_iter=10, nb_epoch=100, scoring='r2', random_state=0, time_dep=False, 
    test_phase=True, d_name=None):

    tested = True
       
    # should check best model is in list of valid models...
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
            if tuning=='rscv':
                params = estimators[
                'SVMReg' if best_model_name=='Bagging_SVMReg' else 
                best_model_name][1]

        tested, _ =  train_test_estimator(
            best_model_name, best_model, X_train, X_test, y_train, y_test,
            scaler=scaler, feat_selector=feat_selector, y_scaler=y_scaler, 
            params=params, tuning=tuning, cv=cv, n_iter=n_iter, 
            scoring=scoring, random_state=random_state, time_dep=time_dep,
            test_phase=test_phase, d_name=d_name)
    else:
        print("Unable to find a 'good-enough' regressor.")
        print("Current regressors suck!")

        tested = False

    return tested


def print_scores(best_model_name, scoring, y_test, y_pred):
    # arguments need check
    print()
    print("[task] === Testing %s" % best_model_name)
    print()
    print(
        "%s optimized for scoring: '%s'" 
        % (best_model_name, scoring.replace('neg_', '')))
    # y_test = np.array([int(yt) for yt in y_test])
    y_test = check_array(y_test, dtype='numeric', ensure_2d=False)
    print("y_test", y_test[0:3])
    print("predictions", y_pred[0:3])
    # non-numerical y objects
    mse = me.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("%s predictions' MSE [%.2f] and RMSE [%.2f] on test data" % (
        best_model_name, mse, rmse))
    r2 = me.r2_score(y_test, y_pred)
    if (y_test > 0).all() and (y_pred > 0).all():
        # you couls use custom_rms_log_err
        rmsle = np.sqrt(me.mean_squared_log_error(y_test, y_pred))
        print("RMSLE on test data: %.2f" % rmsle)
    if (y_test != 0).all():
        # rmspercerr = np.sqrt(np.mean(np.square((y_test - y_pred)/y_test)))
        rmspercerr = reu.custom_rms_perc_err(y_test, y_pred)
        print("RMS Percentage Error on test data: %.2f" % rmspercerr)
    print("R2 score on test data: %.2f" % r2)
    evs = me.explained_variance_score(y_test, y_pred)
    print("Explained Variance score on test data: %.2f." % (evs))


def print_oob_scores(regressor=None, y_train=None):
    if regressor is None:
        raise ValueError("Please provide a valid sklearn regressor.")
    if y_train is None:
        raise ValueError(
                "Please provide a pandas Series or a numpy array")
    
    if hasattr(regressor, 'oob_prediction_'):
        best_oob_prediction = regressor.oob_prediction_
        mse = me.mean_squared_error(y_train, best_oob_prediction)
        rmse = np.sqrt(mse)
        print("oob_predictions' MSE [%.2f] and RMSE [%.2f] on train data." % (
            mse, rmse))
        if (y_train > 0).all() and (best_oob_prediction > 0).all():
            # you couls use custom_rms_log_err
            rmsle = np.sqrt(
                me.mean_squared_log_error(y_train, best_oob_prediction))
            print("oob RMSLE: %.2f" % rmsle)
        if (y_train != 0).all():
            # rmspercerr = np.sqrt(
            #     np.mean(np.square((y_train - best_oob_prediction)/y_train)))
            rmspercerr = reu.custom_rms_perc_err(y_train, best_oob_prediction)
            print("oob RMS Percentage Error: %.2f" % rmspercerr)
        evs = me.explained_variance_score(y_train, best_oob_prediction)
        print("oob Explained Variance score on test data: %.2f" % (evs))
    if hasattr(regressor, 'oob_score_'):
        best_oob_score = regressor.oob_score_
        print("R2 oob_score of refitted ensemble on train data: %1.3f" % 
            best_oob_score)


def train_test_estimator(
        best_model_name, best_model, X_train, X_test, y_train, y_test, 
        scaler=None, y_scaler=None, feat_selector=None, params=None, 
        tuning=None, cv=3, n_iter=10, nb_epoch=100, scoring=None, 
        random_state=0, time_dep=False, test_phase=True, d_name=None):
    """
    Tests optimized or plain estimator [needs refactorization]
    
    ----------------------------------
    test_phase: enables test phase to check best model performance, 
                trains and saves best models if set to False 
                (default: True) 
    """
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if test_phase:
        if hasattr(X_test, 'values'):
            X_test = X_test.values

    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if test_phase:
        if hasattr(y_test, 'values'):
            y_test = y_test.values

    if y_scaler is not None:    
        y_train = y_scaler.inverse_transform(y_train)   
        if test_phase:
            y_test = y_scaler.inverse_transform(y_test)  

    print()
    print("[task] === Training %s" % best_model_name)
    print()

    if best_model_name not in ("Worst", "DummyReg"):
        
        best_model, cv, tuning, params, poly_feats =\
            reu.set_features_params_for_model(
            X_train, time_dep, cv, best_model_name, best_model, tuning, 
            params, random_state, eval_phase=False)

        steps = []
        reg_step = (best_model_name, best_model)
        ppl = None

        if not isinstance(test_phase, bool):
            raise TypeError("'test_phase' should be of type bool")

        if test_phase:
            # assuming data are pre-processed
            pass
        elif not test_phase:
            # train all data and save regressor along w pre-processors
            if scaler is not None:
                steps.append(('scaler', scaler))
            if feat_selector is not None:
                steps.append(('feat_selector', feat_selector))
            if poly_feats is not None:
                steps.append(('poly_features', poly_feats))
        else:
            raise ValueError(
                "'test_phase' should be a bool in [True, False]")

        steps.append(reg_step)
        ppl = Pipeline(steps)

        predicted = None

        if tuning is None and params is None:
            print("Fitting '%s' ..." % best_model_name)

            t0 = time()
            ppl.fit(X_train, y_train)
            t1 = time()

            best_model = ppl
            del ppl

        elif tuning == 'rscv' and params is not None:
            # check params is a dict w tuples (model, distro_params)
            print("Find best params of %s w RSCV and refit it..." % best_model_name)
            # for k, v in params.items():
            #     print("\t", k)
            # print(ppl.get_params().keys())
            # input("press key...")
            try:
                # n_jobs=-2 to avoid burdening your PC
                scorer = reu.get_custom_scorer(scoring, y_train)
                random_search = RandomizedSearchCV(
                    ppl, param_distributions=params, cv=cv, iid=False, 
                    n_iter=n_iter, 
                    scoring=scorer, 
                    refit=True, random_state=random_state, error_score=np.nan)	
            except TypeError as te:
                print(te)
            except Exception as e:
                print(e)
            else:
                t0 = time()
                random_search.fit(X_train, y_train)
                t1 = time()

                # Using the score defined by scoring
                rscv_score = random_search.score(X_train, y_train)
                # mean_rscv_score = random_search.best_score_

                if scoring in (
                    'neg_mean_squared_error', 'neg_rmsle', 'neg_rms_perc_err'):
                    rscv_score = -1.*rscv_score
                    # mean_rscv_score = -1.*mean_rscv_score

                print("Scoring [%s] of refitted estimator on train data: %1.3f"
                    % (scoring.replace('neg_', ''), rscv_score))
                # print("Mean cv score [%s] of the best_estimator: %1.3f"
                #     % (scoring.replace('neg_', ''), mean_rscv_score))

                print("best estimator: \n", random_search.best_estimator_)
                # best_regressor = random_search.best_estimator_.named_steps[
                #     best_model_name.replace("Polynomial", "")]
                best_regressor = random_search.best_estimator_
                # input("Press key to continue...")
                
                print_oob_scores(best_regressor, y_train)

                print()

                # as a check for not fitting the input estimator
                # 
                # best_rscv_params = random_search.best_params_
                # print("Best estimator has params:\n")
                # for param_name in sorted(best_rscv_params.keys()):
                #     print("\t%s: %r" % (param_name, best_rscv_params[param_name]))
                # print()

                # best_model_params = {
                #     k.split('__')[1]: v for k, v in best_rscv_params.items()
                # }

                best_model = best_regressor
                del best_regressor
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

        if test_phase:
            predicted = best_model.predict(X_test)
            print_scores(best_model_name, scoring, y_test, predicted)
            print('Execution time for %s: %.2fs.' % (best_model_name, t1 - t0))
        else:
            print(
                "%s trained and ready to make predictions on unseen data."
                % best_model_name)

            # print("Pipeline:\n")
            # print(ppl)
            # input("Press key to continue...")

            reu.save_regressor(
                best_model, name=best_model_name, d_name=d_name, tuning=tuning)
    else:
        print("Current regressors suck, training not worth the effort :)")

    print()

    return test_phase, best_model