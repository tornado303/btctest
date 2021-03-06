#coding=utf-8
'''
    文件名： btc——predict2
    目标： 使用极值点前20天数据来判断是否是极大极小值，据此来指导买卖
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit=256         #隐层神经元的个数
lstm_layers=2       #隐层层数
#输入数据暂时只有一个，每天的收盘价格
input_size=1
#输出数据为1（极大）或-1（极小）
output_size=1
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
#f=open('btctrain.csv')
df=pd.read_csv('btctrain.csv',header=None)     #读入股票数据
data=df.iloc[:,0:21].values  #取第1-20列 训练数据

'''
def normalized(data_train):
    normalized_train_data = (data_train - np.mean(data_train))/np.std(data_train)
    return normalized_train_data
'''

#获取训练集
def get_train_data(batch_size=60,time_step=1,train_begin=0,train_end=2500):
    batch_index=[]
    data_train = data[train_begin:train_end, :20]
    normalized_train_data=(data_train-np.mean(data_train))/np.std(data_train)  #标准化
    # normalized_train_data = data_train
    #normalized_train_data = normalized(data_train)
    train_x,train_y=[],[]   #训练集

    for i in range(len(data_train)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:20]
       y=data[i:i+time_step, -1, np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())


    batch_index.append((len(data_train)-time_step))
    print('&&&')
    print(train_x[1])
    return batch_index,train_x,train_y


#获取测试集
def get_test_data(time_step=20, test_begin=0):
    data_test=data[test_begin:2500, :20]
    mean=np.mean(data_test)
    std=np.std(data_test)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:20]
       y=data[i*time_step:(i+1)*time_step,-1, np.newaxis]
       test_x.append(x.tolist())
       test_y.extend(y.tolist())
    test_x.append((normalized_test_data[(i+1)*time_step:,:20]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,-1]).tolist())
    return mean,std,test_x,test_y

config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置、dropout参数

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')    
#——————————————————定义神经网络变量——————————————————
def lstmCell():
    #basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm

def lstm(X):
    
    batch_size=tf.shape(X)[0]

    time_step=tf.shape(X)[1]
    print(time_step, "time_step")
    print(batch_size, "batch_size")
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————训练模型————————————————————

def train_lstm(batch_size=60,time_step=1,train_begin=0,train_end=2500):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)

    with tf.variable_scope("sec_lstm"):
        pred, _=lstm(X)
    #print(pred)
    #print('sec_lstm')
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob:0.5})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'model_save3\\modle.ckpt'))
        #I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        #if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")
train_lstm()

#————————————————预测模型————————————————————
def prediction(time_step=1):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean, std, test_x, test_y = get_test_data(time_step)

    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save3')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)):
          prob = sess.run(pred,feed_dict={X:[test_x[step]],keep_prob:1})

          predict=prob.reshape((-1))
          test_predict.extend(predict)


        #print(test_predict)
        #test_y=np.array(test_y)*std[7]+mean[7]
        #test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict)/test_y[:len(test_predict)])  #偏差程度
        print("The accuracy of this predict:",acc)

        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

prediction()
