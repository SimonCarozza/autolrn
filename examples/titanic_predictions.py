from pandas import read_csv
import os

from keras.models import load_model
from sklearn.externals import joblib as jl
import sklearn
import numpy as np
from random import sample

# autolrn classification's module
from autolrn.encoding import labelenc as lc
from autolrn.classification import predict as pr
from pkg_resources import resource_string
from io import StringIO

# Let's make predictions and print their associated probabilities

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# load data
# test data do not have target class 'Survived'

titanic_bytes = resource_string(
            "autolrn", os.path.join("datasets", 'titanic_test.csv'))
titanic_file = StringIO(str(titanic_bytes,'utf-8'))

names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp',
                 'Parch','Ticket','Fare','Cabin','Embarked']

df = read_csv(
    titanic_file, delimiter=",",
    # header=0, names=names,
    na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
    dtype={'Name': 'category', 'Sex': 'category',
           'Ticket': 'category', 'Cabin': 'category',
           'Embarked': 'category'})


# data exploration

print("shape: ", df.shape)

print()


# too many missing values in 'Cabin' columns: about 3/4
print("Dropping 'Cabin' column -- too many missing values")
df.drop(['Cabin'], axis=1, inplace=True)


# replace missing valus in 'Age', 'Embarked'
if df.isnull().values.any():
    print("Null values here... replacing them.")
    df.fillna(method='pad', inplace=True)
    df.fillna(method='bfill', inplace=True)

# print("Original data sample before encoding:", df[:5])

description = df.describe()
print("Description - no encoding:\n", description)

print()
# input("Press key to continue... \n")

original_df = df

original_X = original_df.values

print("Original data sample before encoding (values):", original_X[:5])

df = lc.dummy_encode(df)
df.head()

X = df.values

# print("Original data sample before encoding (values):", original_df[:5])

columns = df.columns

print("Original X shape: ", original_X.shape)
print("Input dim. of original data: %d" % int(original_X.shape[1]))
print("X (encoded) shape: ", X.shape)
print("Input dim. of encoded data: %d" % int(X.shape[1]))
print()

pick_indexes = sample(range(0, len(X)), 10)
print("Indexes to randomly pick new data for predictions:", pick_indexes)
print()

# feat. importance order: sex, age, ticket, fare, name
# Name: %s, sex '%s', age '%d', fare '%.1f'
X_indexes = [2, 3, 4, 7, 8]
feat_str = ("'Name': {}, 'sex': {}, 'age': {:.0f}, 'ticket': {}, 'fare': {:.1f}")
neg = "dead"
pos = "survived"
bin_event = (neg, pos)

clfs = []

try:
    clfs.append(
        jl.load('models/titan_LogRClf_2nd_light_opt_0525.pkl'))
        
    # KerasEstimator = jl.load(
    #     'path/to/feature_transformer.pkl')
    # KerasEstimator.steps.append(
    #     ('KerasClf', load_model('path/to/keras_clf.h5')))
    KerasEstimator = jl.load(
        'models/titan_deep_nn_Clf_2nd_feateng_for_keras_model_0754.pkl')
    KerasEstimator.steps.append(
        ('deep_nn_Clf_2nd_0754', load_model(
            'models/titan_deep_nn_Clf_2nd_None_0754.h5')))
    clfs.append(KerasEstimator)
    KerasEstimator = jl.load(
        'models/titan_KerasClf_2nd_feateng_for_keras_model_0375.pkl')
    KerasEstimator.steps.append(
        ('KerasClf_2nd_refitted_rscv_0375', load_model(
            'models/titan_KerasClf_2nd_refitted_rscv_0375.h5')))
    clfs.append(KerasEstimator)
except FileNotFoundError as fe:
    print(fe)
except Exception:
    raise
else:

    pr.predictions_with_full_estimators(
        clfs, original_X, X, pick_indexes, X_indexes, bin_event, feat_str)
