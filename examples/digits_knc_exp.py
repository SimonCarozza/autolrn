from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from pandas import DataFrame
import numpy as np
from sklearn.datasets import load_digits

import sys
from autolrn import auto_utils as au
from autolrn.classification import eval_utils as eu
from autolrn.classification import param_grids_distros as pgd
from autolrn.classification import train_calibrate as tc
import autolrn.getargs as ga


# starting program
if __name__ == '__main__':

    print()
    print("### Probability Calibration Experiment -- CalibratedClassifierCV "
          "with cv=cv (no prefit) ###")
    print()

    d_name = ga.get_name()

    if d_name is None:
        d_name = 'Digits'

    seed = 7
    np.random.seed(seed)

    # load digits data
    digits = load_digits()

    df = DataFrame(data=digits.data)

    target = 'class'

    df_target = DataFrame(data=digits.target, columns=[target])

    # complete dataframe: features + target

    df[target] = df_target

    if isinstance(df, DataFrame):
        pass
    else:
        raise ValueError("df must be a pandas.DataFrame")

    print(df.shape)

    print("Dataframe description - no encoding:\n", df.describe())
    print()

    print()
    print("=== [task] Train-test split + early pre-processing.")
    print()

    # Missing Attribute Values: None

    auto_feat_eng_data, scoring, Y_type, labels = eu.split_and_X_encode(
        df, target, 0.2, random_state=seed)

    print()

    encoding = auto_feat_eng_data['encoding']
    scaler_tuple = auto_feat_eng_data['scaler']
    featselector = auto_feat_eng_data['feat_selector']
    steps = auto_feat_eng_data['steps']
    X_train_transformed, y_train, X_test_transformed, y_test = auto_feat_eng_data['data_arrays']
    X, y = auto_feat_eng_data['Xy']
    train_index, test_index = auto_feat_eng_data['tt_index']

    n_splits = au.select_nr_of_splits_for_kfold_cv()
    n_iter = au.select_nr_of_iterations()

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ### reproducing the whole autolrn workflow

    names = []
    results = []

    print("Metric:", scoring)
    print("Calibration of untrained models -- CCCV 2nd")
    print()

    # if scoring in ('neg_log_loss', 'brier_score_loss'):
    best_score = 10**4    # 0.0
    best_score_dev = 10**2
    best_cv_results = 10**4*np.ones(n_splits)
    best_exec_time = 31536000    # one year in seconds
    best_model = ('Worst', None, None)

    Dummy_scores = []

    models_data = []
    names = []
    results = []

    scores_of_worst_model = (best_score, best_score_dev, best_cv_results,
                            best_exec_time, best_model)

    scores_of_best_model = scores_of_worst_model

    average_scores_and_best_scores = dict()
    average_scores_and_best_scores['Worst'] \
    = (best_score, best_score_dev, best_exec_time, best_model, {})

    # Start evaluation process

    print()
    print("=== [task] Evaluation of DummyClassifier")
    print()

    wtr = eu.calculate_sample_weight(y_train)

    evaluation_result = eu.single_classic_cv_evaluation(
        X_train_transformed, y_train, 'DummyClf_2nd',
        DummyClassifier(strategy='most_frequent'), wtr, scoring, outer_cv,
        dict(), scores_of_best_model, results, names, seed)

    average_scores_and_best_scores = evaluation_result[0]
    scores_of_best_model = evaluation_result[1]

    Dummy_scores.append(scores_of_best_model[0])    # Dummy score -- ROC_AUC
    Dummy_scores.append(scores_of_best_model[1])    # Dummy score std
    Dummy_scores.append(scores_of_best_model[2])    # Dummy cv results
    Dummy_scores.append(scores_of_best_model[3])    # Dummy execution time
    # Dummy model's name and estimator
    Dummy_scores.append(scores_of_best_model[4])

    names = []
    results = []

    print()

    all_models_and_parameters = dict()

    # Let's assume the best model is GaussianNB 

    print("=== [task] Comparing DummyClassifier to KNeighborsClassifier")
    print()

    evaluation_result = eu.single_nested_rscv_evaluation(
        X_train_transformed, y_train, 'KNeighborsClf_2nd',
        pgd.full_search_models_and_parameters['KNeighborsClf_2nd'][0], 
        pgd.full_search_models_and_parameters['KNeighborsClf_2nd'][1],
        wtr, scoring, n_iter, inner_cv, outer_cv, 
        average_scores_and_best_scores, scores_of_best_model, 
        results, names, seed)

    print()
    au.box_plots_of_models_performance(results, names)

    print()
    print("=== After Non-nested CV evaluation of KNeighborsClassifier...")
    print()

    scores_of_best_model = evaluation_result[1]

    best_model_name = scores_of_best_model[4][0]
    best_model_estim = scores_of_best_model[4][1]

    best_score = scores_of_best_model[0]
    best_score_dev = scores_of_best_model[1]
    best_cv_results = scores_of_best_model[2]
    best_exec_time = scores_of_best_model[3]

    Dummy_score = Dummy_scores[0]
    Dummy_score_dev = Dummy_scores[1]
    Dummy_cv_results = Dummy_scores[2]
    Dummy_exec_time = Dummy_scores[3]

    print()
    print("Currently, best model is '%s' with score '%s': %1.3f (%1.3f)... :" %
          (best_model_name, scoring.strip('neg_'), best_score, best_score_dev))
    print("... execution time: %.2fs" % best_exec_time)
    # print("and prediction confidence: %1.3f" % best_brier_score)
    print()

    if best_model_name != 'DummyClf_2nd':
        # It's assumed best model's performance is
        # satistically better than that of DummyClf on this dataset
        print("DummyClassifier's scores -- '%s': %1.3f (%1.3f)" % (
            scoring.strip('neg_'), Dummy_score, Dummy_score_dev))
        print("'%s' does better than DummyClassifier." % best_model_name)
        if best_exec_time < Dummy_exec_time:
            print("'%s' is quicker than DummyClf." % best_model_name)
        print()
        print()
        input("Press key to continue...")

        preprocessing = (encoding, scaler_tuple, featselector)

        if labels is not None:
            print("You have labels:", labels)
            all_models_and_parameters['labels'] = labels

        print("Defined dictionary with models, parameters and related data.")
        print()

        tc.calibrate_best_model(
            X, y, X_train_transformed, X_test_transformed,
            y_train, y_test, auto_feat_eng_data['tt_index'], 
            preprocessing, scores_of_best_model,
            all_models_and_parameters, n_splits, 0,
            scoring, models_data, d_name, seed)
    else:
        sys.exit("Your best classifier is not a good classifier.")

    input("=== [End Of Program] Enter key to continue... \n")