# Predict International Airline Passengers (t+whatever, given t)
import numpy as np
import pandas
from autolrn.regression.timeseries import ts_utils as tu
from sklearn.metrics import mean_squared_error

# ...

# fix random seed for reproducibility   
np.random.seed(7)

# load the dataset
# names=["Month", "Passengers"]

dataframe = pandas.read_csv(
    'autolrn/datasets/airline-passengers.csv', 
    usecols=[1],  
    # names=names
    )
dataset = dataframe.values
dataset = dataset.astype('float32')

# dataset = dataframe["Passengers"].astype("float32")

train, val, test = tu.train_val_test_split(dataset)

trainPredict, valPredict = np.empty_like(train), np.empty_like(val)
testPredict = np.empty_like(test)

epochs = 200
timesteps = 3
scoring = "mse"
best_model_name = 'Worst'
best_model = None
best_val_score = np.inf
best_train_score = np.inf
# best_reg_std = np.inf

print()
print()
print("*** Best model %s has train score %.3f, val score %.3f [%s]" % (
    best_model_name, best_train_score, best_val_score, scoring))

# best_attr = best_reg_score, best_reg_std, best_model, best_model_name
best_attr = best_train_score, best_val_score, best_model, best_model_name

n_units = 4

names_and_models = tu.create_ts_keras_models(
    n_units=n_units, lag=timesteps, nb_epoch=epochs)

# ... return best stuff
best_attr = tu.evaluate_ts_keras_models(
    names_and_models, dataset, epochs, best_attr, timesteps, scoring)
best_train_score, best_val_score, best_model, best_model_name = best_attr

# compare to naive solution y(t) = y(t+lag) 

if best_model_name not in ('Worst', 'naive_pr'):

    print()
    print("*** Train best model '%s'" % best_model_name)
    print()

    # you may have to reshape data once again here

    # # split into train and test sets
    # train_length = int(len(dataset)*0.75)
    # test_length = len(dataset) - train_length
    # # train, test = dataset[0:train_length,:], dataset[train_length:len(dataset),:]
    # train = dataset[0:train_length]
    train = np.concatenate((train, val), axis=0)

    # test best model
    if best_model_name == "lstm_tstep":
        best_model = tu.names_and_fcts[best_model_name][0](n_units)
    else:
        best_model = tu.names_and_fcts[best_model_name][0](
            n_units, tu.names_and_fcts[best_model_name][1])

    trainPredict, testPredict = tu.test_best_ts_keras_model(
        best_model_name, best_model, train, test, lag=timesteps, nb_epoch=epochs)

    # plot time series for test

    tu.plot_time_series_for_test(
        best_model_name, dataset, trainPredict, testPredict, lag=timesteps)

else:
    print("Unable to find a 'good-enough' ts regressor.")
    print("Current time-series regressors suck!")

print()
input("=== End Of Program")