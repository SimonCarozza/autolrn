# Predict International Airline Passengers (t+whatever, given t)
import numpy as np
import matplotlib.pyplot as plt
import pandas
from autolrn.regression.timeseries import ts_utils as tu
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
names=["Month", "Passengers"]

dataframe = pandas.read_csv(
    'autolrn/datasets/airline-passengers.csv', 
    # header=0,
    usecols=[1], 
    # engine='python',  
    # names=names
    )
dataset = dataframe.values
dataset = dataset.astype('float32')

# dataset = dataframe["Passengers"].astype("float32")

# split into train and test sets
train_size = int(len(dataset)*0.67)
test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
# if train, test are Series
# train, test = dataset.loc[0:train_size], dataset.loc[train_size:len(dataset)]
print("Train and test lenghts:", len(train), len(test))

epochs = 200
timesteps = 3

names_and_models = dict()

n_units = 4
names_and_models["naive_pr"] = (tu.naive_ts_predictor, {})
names_and_models["singlehidden"] = (tu.names_and_fcts["singlehidden"][0](n_units), {})
names_and_models["multilayer"] = (
    tu.names_and_fcts["multilayer"][0](n_units, tu.names_and_fcts["multilayer"][1]), {})

print()
print("Evaluating models...")

for model_name, (model, _) in names_and_models.items():

    lag = timesteps

    print()
    print("model: '%s'..." % model_name)
    print("lag=", 1 if model_name in ("singlehidden", "naive_pr") else lag)
    print()

    trainX, trainY = tu.supervised_timeseries(train, 
        1 if model_name in ("singlehidden", "naive_pr") else lag)
    testX, testY = tu.supervised_timeseries(test, 
        1 if model_name in ("singlehidden", "naive_pr") else lag)

    print("created train and test sets...")

    print("fitting '%s'..." % model_name)
    print()
    if model_name != "naive_pr":
        model.summary()

        model.fit(
        trainX, trainY, epochs=epochs, 
        batch_size=2, verbose=0)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainScore = np.mean(model.evaluate(trainX, trainY, verbose=0))
        testScore = np.mean(model.evaluate(testX, testY, verbose=0))
    else:
        trainPredict = model(trainX)
        testPredict = model(testX)

        trainScore = mean_squared_error(trainX, trainY)
        testScore = mean_squared_error(testX, testY)
    
    print()

    print()
    print("fit completed.")
    print()

    # estimate model performance
    
    print("train score", trainScore)
    print("Train Score for '%s' : %.2f MSE (%.2f RMSE)" % (
        model_name, trainScore, np.sqrt(trainScore)))
    print("Test Score for '%s': %.2f MSE (%.2f RMSE)" % (
        model_name, testScore, np.sqrt(testScore)))

    print()
    print("Plot time series...")
    if model_name in ("singlehidden", "naive_pr"):
        lag = 1

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lag: len(trainPredict) + lag, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(lag*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.title(model_name)
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    print()

    del model

print()