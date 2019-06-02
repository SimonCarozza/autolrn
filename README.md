# autolrn

**autolrn** is a machine learning mini-framework for **automatic tabular data classification and regression** and exploration of the scikit-learn and Keras ecosystems.

autolrn is a humble bunch of small experiments glued together with the ambitious aim of **full automation**, and as such, it's a real mini-framework. The classification module features six sub-modules to preprocess data, train, calibrate, save models and make predictions. The regression module is less rich, but features sub-modules to evaluate and train models - you can complete it by copying code from the classification module.

autolrn keeps an eye on **feature engineering**, so classification methods have been developed to train-test-split data, automatically establish a classification metric - 'roc_auc' vs 'log_loss_score' - based on binary or multiclass target type and get class labels for you to inject your custom hacks without the risk of data leakage from test to train set before Label/One-Hot encoding them separately. 

Regression methods allow you to quickly select whether to label-encode or one-hot encode data for model evaluation with vanilla cross-validation, non-nested RSCV and nested RSCV. You can test the best model accordingly.


## Installation

* Install from Anaconda cloud

Install **python 3.6** first, then install scikit-learn, pandas, matplotlib, keras. If you have [Anaconda Python](https://www.anaconda.com/download/) installed, which I recommend to ease installations in Windows, you can also install py-xgboost, a sklearn wrapper for the popular distributed gradient boosting library [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html), which is optional to autoclf's working.

`conda create env -n alrn python=3.6`

`activate alrn`

`conda install scikit-learn matplotlib pandas keras py-xgboost`

`conda install autolrn`

* Install from Github

First, clone autolrn using git:

`git clone https://github.com/SimonCarozza/autolrn.git`

Then, cd to the autoclf folder and run the install command:

`cd autolrn`
`python setup.py install`

You can go and find pickled models in folder "examples" to make predictions with sklearn 0.20 depending on the version you installe on your PC.


## How to use autolrn's classification module

autolrn's classification module classifies **small and medium datasets** -- from 100 up to 1,000,000 samples -- and makes use of sklearn's jobs to parallelize calculations.

1. Load data as a pandas dataframe, 

   ```python
   import pandas as pd

   df = pd.read_csv(
       'datasets/titanic_train.csv', delimiter=",",
       na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
       dtype={'Name': 'category', 'Sex': 'category',
       'Ticket': 'category', 'Cabin': 'category',
       'Embarked': 'category'})
    ```

1. split data into the usual train-test subsets:

   ```python
   from autolrn.classification import eval_utils as eu

   target = 'Survived'

   sltt = eu.scoring_and_tt_split(df, target, test_size=0.2, random_state=seed)

   X_train, X_test, y_train, y_test = sltt['arrays']
   ```

   1. get scoring, target type and class labels,

   ```python
   scoring = sltt['scoring']
   Y_type = sltt['target_type']
   labels = sltt['labels']
   ```

   1. automatically reduce dataframe to a digestible size (optional)

   ```python
   learnm, ldf = eu.learning_mode(df)
   odf = None

   if len(ldf.index) < len(df.index):

       odf = df
       df = ldf
   ```

1. do your custom feature engineering,

1. automatically (Label / One-Hot) encode subsets and include them into a single dict, 

   ```python
   auto_feat_eng_data = eu.auto_X_encoding(sltt, seed)
   ```

1. pass engineered dict 'sltt' to evaluation method to run a **nested or a classical cross-validation** of estimators with eventual probability calibration.

   ```python
   from autolrn.classification import evaluate as eva
   eva.select_evaluation_strategy(
       auto_feat_eng_data, scoring, Y_type, labels=labels,
       d_name='titanic', random_state=seed, learn=learnm)
   ```

1. make predictions as usual using sklearn's or Keras' predict() method or play with autoclf's predict module.

   ```python
   from pandas import read_csv
   from keras.models import load_model
   from sklearn.externals import joblib as jl
   import numpy as np
   from random import sample

   # auto_learn classfication's module
   from autolrn.encoding import labelenc as lc
   from autolrn.classification import predict as pr

   # ... load test data with read_csv

   original_df = df

   original_X = original_df.values

   # encode the dataframe
   df = lc.dummy_encode(df)

   X = df.values

   pick_indexes = sample(range(0, len(X)), 10)

   # feat. importance order: sex, age, ticket, fare, name
   # Name: %s, sex '%s', age '%d', fare '%.1f'
   X_indexes = [2, 3, 4, 7, 8]
   feat_str = ("'Name': {}, 'sex': {}, 'age': {:.0f}, 'ticket': {}, 'fare': {:.1f}")
   neg = "dead"
   pos = "survived"
   bin_event = (neg, pos)

   clfs.append(
       jl.load('models/titan_AdaBClf.pkl'))
   clfs.append(
       jl.load('models/titan_Bagging_SVMClf.pkl'))

   pr.predictions_with_full_estimators(
       clfs, original_X, X, pick_indexes, X_indexes, bin_event, feat_str)
   ```


## How to use autolrn's regression module

autolrn's regression module can run regressions upon **small and medium datasets** -- from 100 up to 1,000,000 samples -- and makes use of sklearn's jobs to parallelize calculations. 

1. Load data as a pandas dataframe,

   ```python
   df = read_csv(
        "/datasets/rossman_store_train.csv", delimiter=",", 
        parse_dates=['Date'],
        dtype={"StateHoliday": "category"})
   ```

1. split data into the usual train-test subsets:

   ```python
   from autolrn.regression import r_eval_utils as reu

   splitted_data = reu.split_and_encode_Xy(
       X, y, encoding='le', test_size=test_size, shuffle=False)

   X_train, X_test, y_train, y_test = splitted_data["data"]
   # scale target
   tgt_scaler = splitted_data["scalers"][1]
   ```

1. upload estimators and add a KerasRegressor to them:

   ```python
   estimators = dict(pgd.full_search_models_and_parameters)
   ```   

   1. Add a keras regressor to candidate estimators

   ```python
   input_dim = int(X_train.shape[1])
   nb_epoch = au.select_nr_of_iterations('nn')

   keras_reg_name = "KerasReg"

   keras_nn_model, keras_param_grid = reu.create_best_keras_reg_architecture(
       keras_reg_name, input_dim, nb_epoch, pgd.Keras_param_grid)

   estimators[keras_reg_name] = (keras_nn_model, keras_param_grid)
   ```

1. perform model evaluation; in case your data have dates, you can use sklearn's TimeSeriesSplit to run cross-validation to take into account time dependence:

   1. evaluate a baseline - e.g.: evaluate sklearn's DummyRegressor

      ```python
      from sklearn.model_selection import TimeSeriesSplit
      tscv = TimeSeriesSplit(n_splits=2)

      best_attr = eu.evaluate_regressor(
          'DummyReg', DummyRegressor(strategy="median"), 
          X_train, y_train, tscv, scoring, best_attr, time_dep=True)
      ```

   2. select the cross-validation process

      ```python
      from autolrn.regression import evaluate as eu

      # cv_proc in ['cv', 'non_nested', 'nested']
      refit, nested_cv, tuning = eu.select_cv_process(cv_proc='cv')
      ```

   3. compare estimators to baseline

      ```python
      best_model_name, _ , _ , _ = eu.get_best_regressor_attributes(
        X_train, y_train, estimators, best_attr, scoring, 
        refit=refit, nested_cv=nested_cv,
        cv=tscv, time_dep=True, random_state=seed)
      ``` 

3. train and test the best estimator

   ```python
   from autolrn.regression import train as tr

   tr.train_test_process(
      best_model_name, estimators, X_train, X_test, y_train, y_test,
      y_scaler=tgt_scaler, tuning=tuning, cv=tscv, scoring='r2', 
      random_state=seed)
   ```


## DISCLAIMER

autolrn's classification module - which you can download separately as [autoclf](https://github.com/SimonCarozza/autoclf) - is born out of the simple need of **trying out sklearn's and Keras' features**, and as such, it's full of hacks and uses processes that could be replaced with faster solutions. 

autolrn's regression module follows the same philosophy, but performs less tasks more quickly. On the other hand, you can use it to get a taste of time-series analysis.

As a toy concept mini-framework, autolrn **has not been tested following a [TDD](https://en.wikipedia.org/wiki/Test-driven_development) approach**, so it's not guaranteed to be stable and is **not aimed to nor ready for production**.

autolrn has been developed for **Windows 10** but has proved to work smoothly in Ubuntu Linux as well.
