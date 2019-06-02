from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D

import warnings
warnings.filterwarnings("ignore")

### NNs for regression

def baseline_nn_model(input_dim):
    # create model
    bnn_model = Sequential()
    bnn_model.add(Dense(
        units=input_dim, input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    bnn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    bnn_model.compile(
        loss='mse', optimizer='adam', 
        metrics=['mse']
        )
    return bnn_model

def baseline_nn_smaller_model(input_dim):
    # create model
    bnn_model = Sequential()
    bnn_model.add(Dense(
        units=round(input_dim/2), input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    bnn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    bnn_model.compile(
        loss='mse', optimizer='adam', metrics=['mse'])
    return bnn_model

def larger_nn_model(input_dim):
    # create model
    lnn_model = Sequential()
    lnn_model.add(Dense(
        units=input_dim*2, input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    lnn_model.add(Dense(
        units=input_dim, activation='relu', kernel_initializer='normal'))
    lnn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    lnn_model.compile(
        loss='mse', optimizer='adam', metrics=['mse']
     )
    # lnn_model.optimizer.lr = 0.01
    # lnn_model.summary()
    return lnn_model

def deep_nn_model(input_dim):
    # create model
    dnn_model = Sequential()
    dnn_model.add(Dense(
        units=input_dim*5, input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim*5, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    dnn_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # dnn_model.summary()
    return dnn_model

def larger_deep_nn_model(input_dim):
    # create model
    dnn_model = Sequential()
    dnn_model.add(Dense(
        units=input_dim**3, input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim**2, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    dnn_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # dnn_model.optimizer.lr = 0.01
    # dnn_model.summary()
    return dnn_model

def deeper_nn_model(input_dim):
    # create model
    dnn_model = Sequential()
    dnn_model.add(Dense(
        units=input_dim*5, input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim*3, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim*2, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=input_dim, activation='relu', kernel_initializer='normal'))
    dnn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    dnn_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # dnn_model.summary()
    return dnn_model

# eventually, add units_3, drop_out_3
def tunable_deep_nn(
    input_dim=8, units_0=16, units_1=8, units_2=4,
    dropout_rate_0=0.25, dropout_rate_1=0.25, dropout_rate_2=0.1,
    ):
    # create model
    nn_model = Sequential()
    nn_model.add(Dropout(dropout_rate_0))
    nn_model.add(Dense(
        units=units_0, input_dim=input_dim, activation='relu', 
        kernel_initializer='normal'))
    # hidden layers
    nn_model.add(Dropout(dropout_rate_1))
    nn_model.add(Dense(
        units=units_1, activation='relu', kernel_initializer='normal'))
    nn_model.add(Dropout(dropout_rate_2))
    nn_model.add(Dense(
        units=units_2, activation='relu', kernel_initializer='normal'))
    nn_model.add(Dense(
        units=1, kernel_initializer='normal'))
    # compile model
    nn_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # nn_model.optimizer.lr = 0.01
    # nn_model.summary()

    return nn_model

# NNs for time series analysis

# Single Hidden-Layer Perceptron model
def singlelyr_pptron(input_dim=4):
    shlp = Sequential()
    shlp.add(Dense(units=input_dim, 
        input_shape=(1, ),	# (1, )
        activation='relu'))
    shlp.add(Dense(1))
    shlp.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return shlp

# "window" Multilayer Perceptron model 
def multilayer_pptron(input_dim=4, window=3):
    mlp_win = Sequential()
    mlp_win.add(Dense(units=3*input_dim, 
        input_shape=(window, ),
        activation='relu'))
    mlp_win.add(Dense(units=2*input_dim, activation='relu'))
    mlp_win.add(Dense(1))
    mlp_win.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return mlp_win

# previous NNs have no sense of time, RNN have;
# RNNs have a memory of what's been computed by hidden layers

# SimpleRNN w window method
def simple_rnn(input_dim=4, window=3):
    srnn_win = Sequential()
    srnn_win.add(SimpleRNN(units=input_dim, input_shape=(1, window)))
    srnn_win.add(Dense(1))
    srnn_win.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return srnn_win

# for LSTMs, n_units=[4, 8, 12]
# LSTM nn w window method, input_shape=(1, window)
def lstm_win_nn(input_dim=4, window=3):
    lstm_win = Sequential()
    lstm_win.add(LSTM(units=input_dim, 
        input_shape=(1, window)
        ))
    lstm_win.add(Dense(1))
    lstm_win.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return lstm_win

# LSTM nn w timesteps method, input_shape=(None, 1)
def lstm_timesteps_nn(input_dim=4):
    lstm_tstep = Sequential()
    lstm_tstep.add(LSTM(units=input_dim, 
        input_shape=(None, 1)
        ))
    lstm_tstep.add(Dense(1))
    lstm_tstep.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return lstm_tstep

def lstm_memo_nn(input_dim=4, window=3, batch_size=1):
    lstm_memo = Sequential()
    lstm_memo.add(LSTM(
        units=input_dim, 
        batch_input_shape=(batch_size, window, 1), 
        stateful=True
        ))
    lstm_memo.add(Dense(1))
    lstm_memo.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return lstm_memo

def lstm_stc_memo_nn(input_dim=4, window=3, batch_size=1):
    lstm_stc_memo = Sequential()
    lstm_stc_memo.add(LSTM(
        units=input_dim, 
        batch_input_shape=(batch_size, 3, 1), 
        stateful=True, 
        return_sequences=True))
    lstm_stc_memo.add(LSTM(
        units=input_dim,
        stateful=True
        ))
    lstm_stc_memo.add(Dense(1))
    lstm_stc_memo.compile(
        loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return lstm_stc_memo

def gru_win_nn(input_dim=4, window=3):
    gru_win = Sequential()
    gru_win.add(GRU(units=input_dim, input_shape=(1, window)))
    gru_win.add(Dense(1))
    gru_win.compile(
        loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return gru_win

def gru_stc_nn(input_dim=4, window=3):
    gru_stc = Sequential()
    gru_stc.add(GRU(
        units=input_dim, input_shape=(1, window), return_sequences=True))
    gru_stc.add(GRU(units=input_dim, ))
    gru_stc.add(Dense(1))
    gru_stc.compile(
        loss='mse', optimizer='adam', metrics=['mse'])
    return gru_stc

def conv1d_nn(input_dim=4, window=3):
    conv1d = Sequential()
    # conv1d.add(Dense(8,
    # 	input_shape=(3, 1)))
    # conv1d.add(ZeroPadding1D(padding=1))
    # above commented, no padding --> pool_size=1
    conv1d.add(Conv1D(
        filters=input_dim, input_shape=(window, 1),
        kernel_size=window,
        strides=1, use_bias=True, padding='same'))
    conv1d.add(AveragePooling1D(
        pool_size=window, 
        strides=1))
    conv1d.add(Flatten())
    conv1d.add(Dense(1))
    conv1d.compile(
        loss='mean_squared_error', optimizer='adam')
    return conv1d

def conv1d_stc_nn(n_filters=4, window=3):
    conv1d_stc = Sequential()
    conv1d_stc.add(Conv1D(
        filters=n_filters, 
        kernel_size=window, 
        padding='same', strides=1, 
        input_shape=(window, 1)
        ))
    conv1d_stc.add(MaxPooling1D(pool_size=window))
    conv1d_stc.add(Conv1D(
        filters=n_filters, 
        kernel_size=window, 
        padding='same', strides=1, 
        # activation='relu'
        ))
    conv1d_stc.add(MaxPooling1D(pool_size=1))
    conv1d_stc.add(Flatten())
    conv1d_stc.add(Dense(4,))
    conv1d_stc.add(Dense(1,))
    conv1d_stc.compile(
        loss='mse', optimizer='adam', metrics=['mse'])
    return conv1d_stc



