"""Discover where the next crime will be committed in San Francisco."""

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

# from pandas import read_csv
import pandas as pd
import numpy as np

from autolrn import auto_utils as au
from autolrn.classification import eval_utils as eu
from autolrn.classification import evaluate as ev
from autolrn.encoding import labelenc as lc
import autolrn.getargs as ga

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals.joblib.my_exceptions import JoblibValueError

from autolrn.classification import param_grids_distros as pgd
from sklearn.pipeline import Pipeline
from pkg_resources import resource_string
import os
from io import StringIO


def impute_wrong_coords_values_w_medians(df, freqs=None):
    """
    Impute wrong coordinates using medians.

    -----------------------------------------------
    df: X_train dataframe
    freqs: list of frequences of datetime periods - 'Year', 'Month', etc.
    target: label of target single-column dataframe
    """
    df.reset_index(drop=True, inplace=True)
    # print("df index:", df.index)

    print("Dataframe's head:\n", df.head(n=3))

    # medians of X and Y by police district
    listOfPdDistricts = df['PdDistrict'].unique()
    PdDistrictX = (df.groupby('PdDistrict'))['X'].median()
    PdDistrictY = (df.groupby('PdDistrict'))['Y'].median()
    print("Length of 'listOfPdDistricts'", len(listOfPdDistricts))

    # impute wrong values with the medians
    for i in range(len(listOfPdDistricts)):
        df.loc[
            (df['Y'] == 90.0) & (df['PdDistrict'] == listOfPdDistricts[i]),
            'X'] = PdDistrictX[listOfPdDistricts[i]]
        df.loc[
            (df['Y'] == 90.0) & (df['PdDistrict'] == listOfPdDistricts[i]),
            'Y'] = PdDistrictY[listOfPdDistricts[i]]

    print("df.shape after imputing coords values: ", df.shape)
    print()

    if freqs is None:
        freqs = ['Year', 'Month', 'Day', 'hour', 'min']

    df_date_feats = lc.get_date_features(df.copy(), freqs)

    fe_df = lc.get_dummies_or_label_encode(
        df_date_feats.copy(), dummy_cols=200)

    print("fe_df columns:", fe_df.columns)
    print()
    print("Feature dataframe's head:\n", fe_df.head(n=2))

    print()

    return fe_df


# starting program
if __name__ == '__main__':

    print("### Probability Calibration Experiment 'S. Francisco Crime' "
          "-- CalibratedClassifierCV with cv=cv (no prefit) ###")

    seed = 7
    np.random.seed(seed)

    sfc_bytes = resource_string(
        "autolrn", os.path.join("datasets", 'sf_crime_train.csv'))
    sfc_file = StringIO(str(sfc_bytes,'utf-8'))

    df = pd.read_csv(sfc_file, parse_dates=['Dates']).sample(frac=.4)

    ###
    print("Dataframe's shape:", df.shape)
    print()
    print("Dataframe's head:\n", df.head(n=3))
    print()
    print("DataFrame info:\n", df.info())
    print()
    print("Dataframe's unique values:\n", df.nunique())
    print()
    print("Description - no encoding:\n", df.describe())
    print()

    # drop useless categories and
    # those that are difficult to compare with other classes
    df = df[df.Category != 'NON-CRIMINAL']
    df = df[df.Category != 'OTHER OFFENSES']

    # merge 'TREA' crimes with 'TRESPASS'
    df.Category.replace(to_replace='TREA', value='TRESPASS', inplace=True)
    listOfCategories = df.Category.unique()
    assert len(listOfCategories) == 36, \
        "%d!! You have more than 36 categories for your analysis" \
        % len(listOfCategories)

    # reset dataframe index to avoid injecting bogus data
    df.reset_index(drop=True, inplace=True)
    print("df index:\n", df.index)

    # odf = df

    # target class
    target = 'Category'

    print()
    print()
    print("Evaluating models w magic numbers and train-calibrate the best one")

    # df = odf.sample(frac=0.4)

    df.drop(['Descript', 'Resolution'], inplace=True, axis=1)

    sltt = eu.scoring_and_tt_split(df, target, 0.2, seed)

    X_train, X_test, y_train, y_test = sltt['arrays']

    scoring = sltt['scoring']
    Y_type = sltt['target_type']
    labels = sltt['labels']

    freqs = ['Year', 'Month', 'hour', 'min']

    print()
    print("Train feature & target dataframes")

    X_train_imputed =\
    impute_wrong_coords_values_w_medians(X_train.copy(), freqs)
    print()
    print()
    print("Test feature & target dataframes")

    X_test_imputed =\
    impute_wrong_coords_values_w_medians(X_test.copy(), freqs)
    print()

    print("Target train target dataframe's head:\n", y_train.head(n=1))
    print()
    print("Target test target dataframe's head:\n", y_test.head(n=1))

    print()
    print()
    print("Any NaN in train target df 'crime_train' ?",
          y_train.isnull().values.any())
    print("Any NaN in train target df 'crime_test' ?",
          y_test.isnull().values.any())
    print()

    print()
    print("scoring:", scoring)
    print()

    X_test_imputed = eu.reorder_ohencoded_X_test_columns(
        X_train_imputed, X_test_imputed)

    sltt['arrays'] = (X_train_imputed, X_test_imputed, y_train, y_test)

    # end custom feat. engineering

    print("Evaluation, training")
    print()

    # you already did encoding
    auto_feat_eng_data = eu.auto_X_encoding(sltt, random_state=seed, encode=False)

    print()

    ev.perform_classic_cv_evaluation_and_calibration(
        auto_feat_eng_data, scoring, Y_type, labels=labels, random_state=seed)

