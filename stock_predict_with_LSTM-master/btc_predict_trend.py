#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-15
# @Author  : venucia
# @QQ      : 94003990
# @github  : https://github.com/tornado303
# @file    : use trend data to train


from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
CUDA_VISIBLE_DEVICES = 0


#——————————————————导入数据——————————————————————

df = pd.read_csv('traindata1017.csv',header=None)     #读入btc数据
data = df.iloc[:, 0:df.shape[1]].values  #取训练数据

## 网络构建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 16
NUM_EPOCHS = 10
model = Sequential()
model.add(Embedding(8500, EMBEDDING_SIZE,input_length=df.shape[1]-1))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
## 网络训练
# model.save('btcmodel1.h5')
# del model
# model = load_model('btcmodel1.h5')

# Convert labels to categorical one-hot encoding
# one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

def main():
    split = 0.7
    dataX = data[:, 1:]
    dataY = data[:, 0]
    # closedata = data[:, -2]




    # testX = data[5000:6000,0:-2]
    # testY = data[5000:6000,-1]

    # print('*'*20)
    # print(testX,testY)
    # testX = data_test[:,0:-1]
    # testY = data_test[:, -1]
    # 归一化数据
    # scaler = MinMaxScaler()
    # traindataxscaler = scaler.fit_transform(dataX)

    traindataxscaler = dataX

    split_boundary = int(dataX.shape[0] * split)
    train_x =traindataxscaler[: split_boundary]
    validation_x = traindataxscaler[split_boundary:]

    train_y = dataY[: split_boundary]
    validation_y = dataY[split_boundary:]
    # print(train_x, test_x,len(train_x),len(test_x))
    # print(train_y, test_y,len(train_y),len(test_y))
    # print(train_x, train_y)++++++++++++++++++++++++++++++++++++++++++
    # print('train')
    # print(train_x.shape[1],train_y.shape[0])
    ## 网络训练
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(validation_x, validation_y))

    # money = 10000
    # holdingRate = 1
    # fee = 0.0002
    # btcnum = 2
    # k=0
    #
    # listmoney  =[]
    # listbtc = []
    # listbenefit = []
    #
    # for i in range(len(dataX)):
    #    print('-'*30)
    #    xtest = dataX[i].reshape(1, 30)
    #    # if(abs(dataY[i]-model.predict(xtest)) > 0.49):
    #    #     k = k+1
    #    #     print(model.predict(xtest),dataY[i])
    #    #     print(xtest,k/len(dataX),k)
    #    if(model.predict(xtest) > 0.5 and btcnum > 0): #说明下一个位置为极大值，这个时候应该卖
    #         tempmoney = closedata[i] * btcnum *(1-fee)
    #         money = money + tempmoney
    #         btcnum = 0;
    #         print('btcnum:', btcnum, 'cash:', money, 'btcprice:', closedata[i], 'op:sell')
    #         k = k+1
    #         listmoney.append(money)
    #         listbtc.append(btcnum)
    #         listbenefit.append(money + btcnum * closedata[i])
    #    elif(model.predict(xtest)<= 0.5 and money >0): #说明下一个位置为极小值，这个时候应该买
    #         tempBtcNum = money*holdingRate / (closedata[i]*(1+fee))
    #         btcnum = btcnum + tempBtcNum
    #         money = 0
    #         print('btcnum:', btcnum, 'cash:',money, 'btcprice:', closedata[i], 'op:buy')
    #         k = k + 1
    #         listmoney.append(money)
    #         listbtc.append(btcnum)
    #         listbenefit.append(money + btcnum * closedata[i])
    #    else:
    #         print('btcnum:',btcnum, 'cash:',money, 'btcprice:',closedata[i],'op:no')
    #         listmoney.append(money)
    #         listbtc.append(btcnum)
    #         listbenefit.append(money + btcnum * closedata[i])
    #
    # benefitdiff = []
    # for i in range (len(listbenefit)-1):
    #     diff = listbenefit[i+1] - listbenefit[i]
    #     benefitdiff.append(diff)
    #
    # print(benefitdiff)
    #
    #
    #
    # print(listbenefit)
    # plt.plot(listbenefit)
    #
    # sum_benefit = 0
    # profit = 0
    # loss = 0
    # lossnum = 0
    # profitnum = 0
    # losslist = []
    # profitlist = []
    # for i in range (len(benefitdiff)):
    #     sum_benefit += benefitdiff[i]
    #     if (benefitdiff[i] > 0):
    #         profit += benefitdiff[i]
    #         profitnum += 1
    #         profitlist.append(benefitdiff[i])
    #     elif(benefitdiff[i]<0):
    #         loss += benefitdiff[i]
    #         lossnum += 1
    #         losslist.append(benefitdiff[i])
    #
    # print(max(benefitdiff), min(benefitdiff))
    #
    # print('ave_loss:', loss/lossnum, 'ave_profit:', profit/profitnum,'total profit / loss:' ,profit/loss,'benefitdiff:',len(benefitdiff),'listbenefit:',len(listbenefit),profit,loss)









    # plt.plot(closedata)

    

    # for i in range(5):
    #     idx = np.random.randint(len(test_x))
    #     xtest = Xtest[idx].reshape(1,40)
    #     ylabel = ytest[idx]
    #     ypred = model.predict(xtest)[0][0]
    #     sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    #     print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
    #
    # XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    # labels = [int(round(x[0])) for x in model.predict(XX) ]
    # label2word = {1:'积极', 0:'消极'}
    # for i in range(len(INPUT_SENTENCES)):
    #     print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))

main()


