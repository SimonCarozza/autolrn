from sklearn.ensemble import BaggingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_validate, TimeSeriesSplit
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error, r2_score
from time import time
import numpy as np
from . import param_grids_distros as pgd
from . import r_eval_utils as reu
from sklearn.base import is_regressor
from multiprocessing import Pool, cpu_count


def get_data_for_parallel_cv(*args):
    global loop_data
    loop_data = args
    return loop_data

def manual_parallel_cv(cv):
    global loop_data
    try:
        X_train, y_train, regressor, scoring = loop_data
    except NameError as ne:
        print(ne)
    except Exception as e:
        print(e)
    else:
        train_scores = []
        test_scores = []
        train_index, val_index = cv[0], cv[1]
        X_train_tr, X_val = X_train[train_index], X_train[val_index]
        y_train_tr, y_val = y_train[train_index], y_train[val_index]
        regressor.fit(X_train_tr, y_train_tr)
        predicted_train = regressor.predict(X_train_tr)
        predicted_val = regressor.predict(X_val)
        # you can easily implement scores for other regression metrics
        if scoring == "r2":
            train_scores.append(r2_score(y_train_tr, predicted_train))
            test_scores.append(r2_score(y_val, predicted_val))
        elif scoring == "neg_mean_squared_error":
            # mean_squared_error
            train_scores.append(-1*mean_squared_error(y_train_tr, predicted_train))
            test_scores.append(-1*mean_squared_error(y_val, predicted_val))
        elif scoring == "neg_rmsle":
            train_scores.append(-1*reu.custom_rms_log_err(y_train_tr, predicted_train))
            test_scores.append(-1*reu.custom_rms_log_err(y_val, predicted_val))
        elif scoring == "neg_rms_perc_err":
            train_scores.append(-1*reu.custom_rms_perc_err(y_train_tr, predicted_train))
            test_scores.append(-1*reu.custom_rms_perc_err(y_val, predicted_val))
        else:
            # you can easily implement scores for other regression metrics
            raise ValueError("%s is not a valid valued for 'scoring'\n"
                             "valid values are ['r2', 'neg_mean_squared_error', "
                             "'neg_rmsle', 'neg_rms_perc_err']")
    return [train_scores, test_scores]


# if you have evaluation fcts internally define 'refit', 'nested_cv',
# 'tuning' based on cv_proc value, then you can get rid of this method
def select_cv_process(cv_proc='cv'):
    refit, nested_cv, tuning = False, False, None
    if cv_proc=='cv':
        print("*** vanilla cross validation")
    elif cv_proc=='non_nested':
        print("*** non-nested randomized-search cv")
        refit, nested_cv, tuning = True, False, 'rscv'
    elif cv_proc=='nested':
        print("*** nested [rs]cross-validation")
        refit, nested_cv, tuning = True, True, 'rscv'
    else:
        raise ValueError("'%s' is not a valid value for 'cv_prov',\n"
                         "valid values are ['cv', 'non_nested', 'nested']"
                         % cv_proc)

    return refit, nested_cv, tuning


# if you're gonna use these methods for keras models,
# make sure you're passing KerasRegressor instances

def evaluate_regressor(
    name, regressor, X_train, y_train, cv=3, scoring=None, best_attr=None,
    time_dep=False, refit=False, nested_cv=False, results=None):

    if scoring not in (
        "neg_rmsle", "neg_rms_perc_err", "neg_mean_squared_error", "r2"):
        if scoring is None:
            raise ValueError("You should provide a metric for scoring.")
        else:
            raise ValueError(
        "%s is not a valid value for 'scoring', valid values are "
        "['neg_rmsle', 'neg_rms_perc_err', 'neg_mean_squared_error', 'r2']"
        % scoring)

    best_model_name, best_model, best_reg_score, best_reg_std = best_attr

    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    print("Evaluating %s ..." % name)

    # print("Estimator", estimator)
    if not time_dep:
        if y_train is not None and name == 'KNeighborsReg':
            y_train = check_array(y_train, dtype='numeric', ensure_2d=False)
    else:
        if not isinstance(cv, TimeSeriesSplit):
            raise TypeError(
                        "'%r' is not a valid type for time series splitting\n"
                        "valid type: 'sklearn.model_selection.TimeSeriesSplit'"
                        % type(cv))
        else:
            if isinstance(regressor, RandomizedSearchCV):
                pass
            else:
                if not is_regressor(regressor):
                    # keras.wrappers.scikit_learn.KerasRegressor
                    if not isinstance(regressor, KerasRegressor):
                        print("Oops, %r type." % regressor)
                        raise TypeError(
                            "non-sklearn regressor should be of type KerasRegressor")

    train_score, train_score_std, score_std, score = 0., 0., 0., 0.
    exec_time = 0.
    success = 0
    try:
        # nested cv and vanilla cv, respectively
        # fitting a RSCV object or a model
        if (refit and nested_cv) or (not refit and not nested_cv):

            scorer = reu.get_custom_scorer(scoring, y_train)
            # if (y_train < 0).any():
            #     print("y_train has some negative values")
            t0 = time()
            results = cross_validate(
                regressor, X_train, y_train, cv=cv, n_jobs=-1,
                return_train_score=True,
                pre_dispatch='2*n_jobs',
                scoring=scorer,
                error_score='raise'
                )
            t1 = time()
        else:
            # refit=True, nested_cv=None
            # get results from RSCV
            if results is None:
                raise ValueError("None is not a valid value for results")
            else:

                t0 = time()
                results = {
                    "train_score": results["mean_train_score"],
                    "std_train_score": results["std_train_score"],
                    "test_score": results["mean_test_score"],
                    "std_test_score": results["std_test_score"],
                    "total_time": results["total_time"]}
                t1 = time()

    except MemoryError as me:
        t1 = time()
        print('%s -- MemoryError. Total execution time: %.2fs.' % (name, t1 - t0))
        print(me)
    except ValueError as ve:
        t1 = time()
        print('%s -- ValueError. Total execution time: %.2fs.' % (name, t1 - t0))
        print(ve)
    except TypeError as te:
        t1 = time()
        print('TypeError for %s. Total execution time: %.2fs.' % (name, t1 - t0))
        print(te)
    except NameError as ne:
        t1 = time()
        print('NameError for %s. Total execution time: %.2fs.' % (name, t1 - t0))
        print(ne)
    except Exception as e:
        print(e)
        # raise e
        t1 = time()
        print('%s -- uncaught exc. -- Total execution time: %.2fs.' % (name, t1 - t0))
    else:
        success = 1
        train_score = np.mean(results["train_score"])
        score = np.mean(results["test_score"])
        if refit and not nested_cv:
            # train_score = np.mean(results["train_score"])
            train_score_std = np.mean(results["std_train_score"])
            # score = np.mean(results["test_score"])
            score_std = np.mean(results["std_test_score"])
            exec_time = results["total_time"]
        else:
            # train_score = np.mean(results["train_score"])
            train_score_std = np.std(results["train_score"])
            # score = np.mean(results["test_score"])
            score_std = np.std(results["test_score"])
            exec_time = t1-t0

    if success:
        print("Mean train score [%s] for %s: %.3f (%.3f)" % (
            scoring, name, train_score, train_score_std))
        print("Mean cross val. score [%s] for %s: %.3f (%.3f)" % (
            scoring, name, score, score_std))
        if scoring == 'neg_mean_squared_error':
            rmse = np.sqrt(-1*score)
            rmse_std = np.sqrt(score_std)
            print("Root MSE %.3f (%.3f)" % (rmse, rmse_std))
        print('Execution time for %s: %.2fs w cross validation.' % (name, exec_time))

        if score > best_reg_score:
            best_reg_score = score
            best_reg_std = score_std
            best_model_name = name
            if isinstance(regressor, RandomizedSearchCV):
                # print(regressor)
                best_model = regressor.get_params()[
                    "estimator__" + best_model_name.replace("Polynomial", "")]
            else:
                best_model = regressor
            print(
                "*** Best model %s has score [%s] %.3f +/- %.3f" % (
                    best_model_name, scoring, best_reg_score, best_reg_std))

        best_attr = best_model_name, best_model, best_reg_score, best_reg_std  

    print()

    return best_attr


def get_best_regressor_attributes(
    X_train, y_train, estimators, best_attr=None, scoring=None,
    nested_cv=False, cv=3, tuning='rscv', n_iter=10, refit=True,
    time_dep=False, random_state=0):

    if scoring not in (
        "neg_rmsle", "neg_rms_perc_err", "neg_mean_squared_error", "r2"):
        if scoring is None:
            raise ValueError("You should provide a metric for scoring.")
        else:
            raise ValueError(
        "%s is not a valid value for 'scoring', valid values are "
        "['neg_rmsle', 'neg_rms_perc_err', 'neg_mean_squared_error', 'r2']"
        % scoring)

    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    if (y_train < 0).any():
        print("y_train has some negative values")

    print()
    if not refit and not nested_cv:
        print("*** Evaluate optimized / default models w cross validation...")
    elif refit and not nested_cv:
        print("*** Find hyperparameters w rscv, refit and score best regressors.")
    elif not refit and nested_cv:
        print("*** nested_cv=True requires 'refit'=True... done!")
        refit = True
    else:
        # if refit and nested_cv:
        print("*** Nested rscv: evaluate rscv objects w cv")
    print()

    if "RidgeReg" in estimators.keys():
        if X_train.shape[1] < 15 and X_train.shape[0] < 500000:
            print("Found RidgeReg model, creating a polynomial out of it...")
            poly_dict = {"PolynomialRidgeReg": (
                estimators["RidgeReg"][0], estimators["RidgeReg"][1]
                )}
            poly_dict.update(estimators)
            estimators = poly_dict
            # print(estimators.keys())

    X_train_temp = None

    for name, (model, params) in estimators.items():
        try:
            model, cv, tuning, params, poly_feats =\
            reu.set_features_params_for_model(
            X_train, time_dep, cv, name, model, tuning, params, random_state)
        except MemoryError as me:
            continue
        except Exception as e:
            print(e)
            continue

        if poly_feats is not None:
            X_train_temp = X_train
            X_train = poly_feats.transform(X_train)
        else:
            # print("No poly_features")
            pass

        results = None
        success = 1
        if refit:
            ppl = Pipeline([(name.replace("Polynomial", ""), model)])
            try:
                success = 0

                scorer = reu.get_custom_scorer(scoring, y_train)
                random_search = RandomizedSearchCV(
                    ppl, param_distributions=params, cv=cv, iid=False,
                    n_iter=n_iter,
                    scoring=scorer,
                    refit=refit,
                    # verbose=2,
                    random_state=random_state, error_score=np.nan)
            except AttributeError as ae:
                print(ae)
            except MemoryError as me:
                print(me)
            except TypeError as te:
                print(te)
            except Exception as e:
                print(e)
            else:
                if not nested_cv:
                    # print("Running non-nested cv...")
                    try:
                        random_search.fit(X_train, y_train)
                    except ValueError as ve:
                        print(ve)
                        print("Unable to run non-nested rscv with", name)
                        print()
                        continue
                    except Exception as e:
                        print(e)
                    else:
                        success = 1
                        model = random_search.best_estimator_.get_params()[
                            name.replace("Polynomial", "")]
                        results = random_search.cv_results_
                        results["total_time"] = np.sum(results["mean_fit_time"])
                        + random_search.refit_time_
                else:
                    success = 1
                    # nested_cv = True
                    # print("Running nested cv...")
                    model = random_search
                    # print()
                    # print("RSCV object:\n", model)

        if success:
            best_attr = evaluate_regressor(
                name, model, X_train, y_train, cv, scoring, best_attr, time_dep,
                refit, nested_cv, results)

        if name == "PolynomialRidgeReg":
            X_train = X_train_temp
            del X_train_temp

    return best_attr
