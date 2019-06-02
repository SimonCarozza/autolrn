"""
timeseries submodule: delivers basic single-step forecasts.
"""
from .. import neuralnets as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# results depends on how many timesteps/lags have been taken
def naive_ts_solution(datum):
    return datum

def naive_ts_predictor(test_data):
    predictions = []
    for v in test_data:
        predictions.append(naive_ts_solution(v))
    return np.array(predictions)


# each dict element k:v --> name: {build_fn, lag, input_shape}
names_and_fcts = {
    # "naive_pr": (naive_ts_predictor, 0),
    "singlehidden": (nn.singlelyr_pptron, 1), 
    "multilayer": (nn.multilayer_pptron, 3),
    "srnn_win": (nn.simple_rnn, 3),
    "lstm_win": (nn.lstm_win_nn, 3),
    "lstm_tstep": (nn.lstm_timesteps_nn, 3),
    "lstm_memo": (nn.lstm_memo_nn, 3),
    "lstm_stc_memo": (nn.lstm_stc_memo_nn, 3),
    "gru_win": (nn.gru_win_nn, 3),
    "gru_stc_win": (nn.gru_stc_nn, 3),
    "conv1d": (nn.conv1d_nn, 3),
    "conv1d_stc": (nn.conv1d_stc_nn, 3),
}


def create_ts_keras_models(
    names_and_fcts=None, n_units=4, lag=1, nb_epoch=1, batch_size=1, 
    k_regressors=False):

    print()
    print("lag: %d timesteps" % lag)

    if names_and_fcts is None:
        names_and_fcts = {
            "srnn_win": (nn.simple_rnn, lag),
            "lstm_win": (nn.lstm_win_nn, lag),
            "lstm_tstep": (nn.lstm_timesteps_nn, lag),
            "lstm_memo": (nn.lstm_memo_nn, lag),
            "lstm_stc_memo": (nn.lstm_stc_memo_nn, lag),
            "gru_win": (nn.gru_win_nn, lag),
            "gru_stc_win": (nn.gru_stc_nn, lag),
            "conv1d": (nn.conv1d_nn, lag),
            "conv1d_stc": (nn.conv1d_stc_nn, lag)
        }
    else:
        # check each models_dict element has a name key and 
        # a value whose first element is a build_fn... 
        pass

    names_and_models = {"naive_pr": (naive_ts_predictor, 0)}

    if type(k_regressors) is bool:

        for k, v in names_and_fcts.items():
            # this way you can pass non legal sk_params right to build_fn
            if k in ("singlehidden", "lstm_tstep"):
                names_and_models[k] = (v[0](n_units), {})
            else:
                names_and_models[k] = (v[0](n_units, v[1]), {})

        if k_regressors:
            keras_Reg_fcts = names_and_models

            # if input_dim < 15:
            #     keras_Reg_fcts['larger_deep_nn'] = (None, {})
            
            for k, v in keras_Reg_fcts.items():
                if k != "naive_pr":
                    names_and_models[k] = (KerasRegressor(
                        build_fn=v[0], nb_epoch=nb_epoch,
                        # input_dim=n_units, 
                        batch_size=batch_size,
                        verbose=0), {})

    else:
        raise ValueError(
            "'%s' is not a valid value for 'k_regressors'\n"
            "valid values are of type bool [True, False]")
    
    return names_and_models


def create_best_ts_keras_regressor(keras_tsk_name, input_shape, nb_epoch, keras_param_grid):
    pass

# the following code is adapted from Dr. Jason Brownlee's 2016 work
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/,
# which has updated examples indeed


# convert an array of values into a dataset matrix
def supervised_timeseries(dataset, lag=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-lag-1):
		dataX.append(dataset[i:(i+lag), 0])
		dataY.append(dataset[i+lag, 0])
	return np.array(dataX), np.array(dataY)


def train_val_test_split(dataset, train_size=0.75):
    # split into train and test sets
    train_length = int(len(dataset)*train_size)
    test_length = len(dataset) - train_length
    # train, test = dataset[0:train_length,:], dataset[train_length:len(dataset),:]
    train, test = dataset[0:train_length], dataset[train_length:len(dataset)]

    # create a validation set out of the train set --> 50/25/25
    train_length = int(len(train)*train_size)
    train, val = train[0: train_length], train[train_length: len(train)]
    # if train, test are Series
    # train, test = dataset.loc[0:train_length], dataset.loc[train_length:len(dataset)]
    print("Train, val, test lenghts:", len(train), len(val), len(test))

    return train, val, test


def reshape_train_test(model_name, train, test, lag=1):

    trainX, trainY = supervised_timeseries(train, lag)		
    testX, testY = supervised_timeseries(test, lag)

    print("Before reshaping:")
    print("shape of train arrays (X, Y):", trainX.shape, trainY.shape)
    print("shape of test arrays (X, Y):", testX.shape, testY.shape)

    if model_name in ("srnn_win", "lstm_win", "gru_win", "gru_stc_win"):
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # input_shape = (1, lag)
    elif model_name in (
        "lstm_tstep", "lstm_memo", "lstm_stc_memo", "conv1d", "conv1d_stc"):
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        # input_shape = (None, 1)

    print("... after reshaping:")
    print("shape of train arrays (X, Y):", trainX.shape, trainY.shape)
    print("shape of test arrays (X, Y):", testX.shape, testY.shape)

    return trainX, testX, trainY, testY


def evaluate_ts_keras_models(
    models_dict, dataset, nb_epoch, best_attr, lag=1, scoring="mse"):

    best_train_score, best_val_score, best_model, best_model_name = best_attr

    print()
    print("Evaluating models...")

    # normalize the dataset - LTSMs are sensitive!

    train, val, _ = train_val_test_split(dataset)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    # print(len(train), len(val))

    for model_name, (model, _) in models_dict.items():

        print()
        print("model: '%s'..." % model_name)
        print("lag=", 1 if model_name in ("singlehidden", "naive_pr") else lag)
        print()

        trainX, valX, trainY, valY = reshape_train_test(model_name, train, val, lag)

        # print()
        # input("Press key to continue...")

        print("created train and val sets...")

        print("fitting '%s'..." % model_name)
        print()
        if model_name != "naive_pr":
            model.summary()
        print()

        if model_name == "naive_pr":
            # generate predictions
            trainPredict = model(trainX)
            valPredict = model(valX)
            print("Naive train and val predictions:", len(trainPredict), len(valPredict))

        elif model_name not in ("lstm_memo", "lstm_stc_memo"):
            model.fit(
                trainX, trainY, epochs=nb_epoch, 
                batch_size=2, verbose=0)

            # generate predictions
            trainPredict = model.predict(trainX)
            valPredict = model.predict(valX)
        else:
            print()
            for i in range(nb_epoch):
                model.fit(
                    trainX, trainY, epochs=1, batch_size=1, verbose=0, shuffle=False
                    )
                model.reset_states()

            trainPredict = model.predict(trainX, batch_size=1)
            model.reset_states()
            valPredict = model.predict(valX, batch_size=1)

        print()
        print("fit completed.")
        print()

        # invert predictions to have the same units as the original data
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        valPredict = scaler.inverse_transform(valPredict)
        valY = scaler.inverse_transform([valY])

        if scoring=="mse":
            trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
            rTrainScore = np.sqrt(trainScore)
            valScore = mean_squared_error(valY[0], valPredict[:, 0])
            rvalScore = np.sqrt(valScore)
            print("Train Score for '%s': %.2f MSE (%.2f RMSE)" % (
                model_name, trainScore, rTrainScore))
            print("val Score for '%s': %.2f MSE (%.2f RMSE)" % (
                model_name, valScore, rvalScore))
        else:
            # replace 'other_metric' with metric  of your choice
            # trainScore = other_metric(trainY[0], trainPredict[:, 0])
            # print("Train Score for '%s': %.2f MSE (%.2f RMSE)" % (
            #     model_name, trainScore))
            # valScore = other_metric(valY[0], valPredict[:, 0])
            # print("val Score for '%s': %.2f MSE" % (model_name, valScore))
            pass
        print("* Best ts score for '%s': %.2f" % (best_model_name, best_val_score))

        dataset = scaler.inverse_transform(dataset)

        if valScore < best_val_score:
            best_train_score = trainScore
            best_val_score = valScore
            # best_reg_std = score_std
            best_model = model
            best_model_name = model_name
            print(
                "*** Best model %s has train score %.3f, val score %.3f [%s]" % (
                    best_model_name, best_train_score, best_val_score, scoring))

        best_attr = best_train_score, best_val_score, best_model, best_model_name

        print()

        del model

    return best_attr


def test_best_ts_keras_model(best_model_name, best_model, train, test, lag=1, nb_epoch=1):

    # normalize the dataset - LTSMs are sensitive!
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    trainX, testX, trainY, testY = reshape_train_test(
        best_model_name, train, test, lag)

    if best_model_name not in ("lstm_memo", "lstm_stc_memo"):
        best_model.fit(
            trainX, trainY, epochs=nb_epoch, 
            batch_size=2, verbose=0)

        # generate predictions
        trainPredict = best_model.predict(trainX)
        testPredict = best_model.predict(testX)
    else:
        print()
        for i in range(epochs):
            best_model.fit(
                trainX, trainY, epochs=1, batch_size=1, verbose=0, shuffle=False
                )
            best_model.reset_states()

        trainPredict = best_model.predict(trainX, batch_size=1)
        best_model.reset_states()
        testPredict = best_model.predict(testX, batch_size=1)

    # invert predictions to have the same units as the original data
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
    # trainScore = np.mean(
    #     model.evaluate(trainY[0], trainPredict[:, 0], verbose=0))
    rTrainScore = np.sqrt(trainScore)
    print("Best ts train Score for '%s': %.2f MSE (%.2f RMSE)" % (
        best_model_name, trainScore, rTrainScore))
    testScore = mean_squared_error(testY[0], testPredict[:, 0])
    # testScore = np.mean(testY[0], testPredict[:, 0], verbose=0)
    rtestScore = np.sqrt(testScore)
    print("*** Best ts test Score for '%s': %.2f MSE (%.2f RMSE)" % (
        best_model_name, testScore, rtestScore))

    return trainPredict, testPredict


def plot_time_series_for_test(model_name, dataset, trainPredict, testPredict, lag=1):
    print()
    print("Plot best time series...")

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

    return plt