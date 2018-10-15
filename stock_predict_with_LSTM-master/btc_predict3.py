#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
#from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
#import matplotlib
import tushare as ts

import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import datetime
from pandas import DataFrame
from numpy import row_stack,column_stack
import pandas

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''
rnn_unit=30         #隐层神经元的个数
lstm_layers=2       #隐层层数
input_size=20
output_size=1
lr=0.0006         #学习率

'''


#——————————————————导入数据——————————————————————
#f=open('btctrain.csv')
df=pd.read_csv('btctrain.csv',header=None)     #读入股票数据
data=df.iloc[:,0:21].values  #取第1-20列 训练数据

#scaler = MinMaxScaler()
#data = scaler.fit_transform(data)

#print(data_label)

#配置gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#——————————————————定义神经网络变量——————————————————

def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=20, return_sequences=True))
    #model.add(LSTM(6, input_dim=1, return_sequences=True))
    #model.add(LSTM(6, input_shape=(None, 1),return_sequences=True))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(keras.layers.Dropout(0.5))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=50, nb_epoch=60, validation_split=0.1)
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print(predict)
    print(test_y)
    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
       print(e)
    return predict, test_y


def main():
    #train_x, train_y, test_x, test_y, scaler = load_data('international-airline-passengers.csv')
    split = 0.8
    dataX = data[:,0:-1]
    dataY = data[:,-1]
    traindatax = dataX.reshape(-1,1)

    # 归一化数据
    scaler = MinMaxScaler()
    traindataxscaler = scaler.fit_transform(traindatax)

    dataA = []
    for i in range(len(traindataxscaler) // 20 - 1):
        dataA.append(traindataxscaler[i*20: i*20 + 20 + 1])

    reshaped_dataX = np.array(dataA).astype('float64')
    reshaped_dataY = dataY.reshape(-1, 1)

    split_boundary = int(reshaped_dataX.shape[0] * split)
    train_x =reshaped_dataX[: split_boundary]
    test_x = reshaped_dataY[split_boundary:]

    train_y = reshaped_dataY[: split_boundary]
    test_y = reshaped_dataY[split_boundary:]

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)

    '''
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    test_y = scaler.inverse_transform(test_y)
    '''
    # fig2 = plt.figure(2)
    # plt.plot(predict_y, 'g:')
    # plt.plot(test_y, 'r-')
    # plt.show()

main()
