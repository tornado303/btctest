#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

#参数设置
#——————————————————参数设置——————————————————————
lr = 1e-3
input_size = 28      # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
timestep_size = 28   # 时序持续长度为28，即每做一次预测，需要先输入28行
hidden_size = 256    # 隐含层的数量
layer_num = 2        # LSTM layer 的层数
class_num = 10       # 最后输出分类类别数量，如果是回归预测的话应该是 1
cell_type = "lstm"   # lstm 或者 block_lstm

X_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32, [None, class_num])
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32, batch_size = 128
keep_prob = tf.placeholder(tf.float32, [])


#——————————————————导入数据——————————————————————
#f=open('btctrain.csv')
df=pd.read_csv('traindata1.csv',header=None)     #读入股票数据
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

#配置gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#——————————————————定义神经网络变量——————————————————

#——————————————————定义神经网络变量——————————————————
# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键

# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(X_input, [-1, timestep_size, input_size])

# ** 步骤2：创建 lstm 结构
def lstm_cell(cell_type, num_nodes, keep_prob):
    assert(cell_type in ["lstm", "block_lstm"], "Wrong cell type.")
    if cell_type == "lstm":
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(num_nodes)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(cell_type, hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)

# **步骤3：用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# **步骤4：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size],
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# h_state = state[-1][1]

# # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
# # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数，我们可以用它来展开实现LSTM按时间步迭代。
# # **步骤4：方法二，按时间步展开计算
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        (cell_output, state) = mlstm_cell(X[:, timestep, :],state)
        outputs.append(cell_output)
h_state = outputs[-1]

#————————————————训练模型————————————————————

############################################################################
# 以下部分其实和之前写的多层 CNNs 来实现 MNIST 分类是一样的。
# 只是在测试的时候也要设置一样的 batch_size.

# 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
# 首先定义 softmax 的连接权重矩阵和偏置

import time

# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)


# 损失和评估函数
cross_entropy = -tf.reduce_mean(y_input * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.global_variables_initializer())
time0 = time.time()

'''
            for i in range(100):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob:0.5})
    '''

for i in range(5000):
    _batch_size=100
    X_batch, y_batch = mnist.train.next_batch(batch_size=_batch_size)
    cost, acc,  _ = sess.run([cross_entropy, accuracy, train_op], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 0.5, batch_size: _batch_size})
    if (i+1) % 500 == 0:
        # 分 100 个batch 迭代
        test_acc = 0.0
        test_cost = 0.0
        N = 100
        for j in range(N):
            X_batch, y_batch = mnist.test.next_batch(batch_size=_batch_size)
            _cost, _acc = sess.run([cross_entropy, accuracy], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 1.0, batch_size: _batch_size})
            test_acc += _acc
            test_cost += _cost
        print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}; pass {}s".format(i+1, cost, acc, test_cost/N, test_acc/N, time.time() - time0))
        time0 = time.time()

#————————————————预测模型————————————————————
'''
def prediction(time_step=1):
    X=tf.placeholder(tf.float32, shape=[1,time_step,input_size])
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
'''