from sklearn.ensemble import BaggingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_validate, TimeSeriesSplit
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error, r2_score
from time import time
import numpy as np
from . import param_grids_distros as pgd
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
            train_scores.append(-1+mean_squared_error(y_train_tr, predicted_train))
            test_scores.append(-1*mean_squared_error(y_val, predicted_val))
        else:
            # you can easily implement scores for other regression metrics
            raise ValueError("%s is not a valid valued for 'scoring'\n"
                             "valid values are ['r2', 'neg_mean_squared_error']")
    return [train_scores, test_scores]


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
    best_model_name, best_model, best_reg_score, best_reg_std = best_attr
    
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
    try:
            
        # nested cv and vanilla cv, respectively
        # fitting a RSCV object or a model
        if (refit and nested_cv) or (not refit and not nested_cv):
            # n_jobs=-2 to avoid burdening your PC
            t0 = time()
            results = cross_validate(
                regressor, X_train, y_train, cv=cv, n_jobs=-1,
                return_train_score=True, 
                pre_dispatch='2*n_jobs', scoring=scoring)
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
        if refit and not nested_cv:
            train_score = np.mean(results["train_score"])
            train_score_std = np.mean(results["std_train_score"])
            score = np.mean(results["test_score"])
            score_std = np.mean(results["std_test_score"])
            exec_time = results["total_time"]
        else:
            train_score = np.mean(results["train_score"])
            train_score_std = np.std(results["train_score"])
            score = np.mean(results["test_score"])
            score_std = np.std(results["test_score"])
            exec_time = t1-t0

    print("Mean train score [%s] for %s: %.3f (%.3f)" % (
        scoring, name, train_score, train_score_std))
    print("Mean cross val. score [%s] for %s: %.3f (%.3f)" % (
        scoring, name, score, score_std))
    if scoring == 'neg_mean_squared_error':
        rmse = np.sqrt(score)
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
                "estimator__" + best_model_name.lstrip("Polynomial")]
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
    nested_cv=False, cv=3, n_iter=10, refit=True, time_dep=False, 
    random_state=0):

    print()
    if not refit and not nested_cv:
        print("*** Evaluate optimized / default models w cross validation.")
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

    for name, (model, params) in estimators.items():            
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

        if name == "PolynomialRidgeReg":
            interact_only=False
            if X_train.shape[0] > 100000:
                interact_only=True
            poly_features = PolynomialFeatures(
                degree=5, interaction_only=interact_only)
            X_train_temp = X_train.copy()
            try:
                poly_features.fit_transform(X_train)
            except MemoryError as me:
                print("MemoryError -- Unable to creare polynomial features.")
                print(me)
                continue
            except Exception as e:
                raise e
            else:
                X_train = poly_features.fit_transform(X_train)
                
        elif name == 'SVMReg' and len(y_train) > 10000:
                name = 'Bagging_SVMReg'
                print("*** SVR detected...")

                model.set_params(kernel='linear')

                n_estimators = 5
                bagging = BaggingRegressor(
                    model, max_samples=1.0/n_estimators, n_estimators=n_estimators,
                    random_state=random_state)

                model = bagging

                params = {
                    name + '__' +
                    k: v for k, v in pgd.Bagging_param_grid.items()}

        results=None
        if refit:
            ppl = Pipeline([(name.lstrip('Polynomial'), model)])
            try:
                regressor = RandomizedSearchCV(
                    ppl, param_distributions=params, cv=cv, iid=False, 
                    n_iter=n_iter, scoring=scoring, refit=refit, 
                    # verbose=2,
                    random_state=random_state, error_score=np.nan)
            except AttributeError as ae:
                print(ae)
            except TypeError as te:
                print(te)
            except Exception as e:
                print(e)
            else:
                if not nested_cv:
                    regressor.fit(X_train, y_train)
                    model = regressor.best_estimator_.get_params()[
                        name.lstrip('Polynomial')]
                    results = regressor.cv_results_
                    results["total_time"] = np.sum(results["mean_fit_time"]) 
                    + regressor.refit_time_
                else: 
                    # nested_cv = True
                    model = regressor
                    # print()
                    # print("RSCV object:\n", model)

        best_attr = evaluate_regressor(
            name, model, X_train, y_train, cv, scoring, best_attr, time_dep, 
            refit, nested_cv, results)
        # print()

        if name == "PolynomialRidgeReg":
            X_train = X_train_temp
            del X_train_temp

    return best_attr