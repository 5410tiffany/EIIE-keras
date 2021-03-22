# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:05:25 2020

@author: Joukey
"""

from pgportfolio.marketdata.datamatrices import DataMatrices
from pgportfolio.tools.configprocess import load_config
import numpy as np
import tensorflow as tf
import tflearn
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3     #占用30%显存

np.random.seed(0)
tf.set_random_seed(0)
sess = tf.Session(config=tf_config)

def max_drawdown(pv_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    drawdown_list = []
    max_benefit = 0
    for i in range(pv_array.shape[0]):
        if pv_array[i] > max_benefit:
            max_benefit = pv_array[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - pv_array[i] / max_benefit)
    return max(drawdown_list)

def port_value(pv_array):
    p = np.array([np.prod(pv_array[:i+1]) for i in range(pv_array.shape[0])])
    return p

#%% load config & data

config_2 = load_config(2)
# config_2['input']['start_date'] = '2015/05/01'  
# config_2['input']['end_date'] = '2017/04/27'    # 自訂區間
data = DataMatrices.create_from_config(config_2)

#%% network structure

feature_number = config_2["input"]["feature_number"]
rows = config_2["input"]["coin_number"]
columns = config_2["input"]["window_size"]
input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns]) #[None, 3, 11, 31]
previous_w = tf.placeholder(tf.float32, shape=[None, rows]) #[None, 11]
input_num = tf.placeholder(tf.int32, shape=[]) #[11]
y = tf.placeholder(tf.float32, shape=[None, feature_number, rows])

def allint(l):
    return [int(i) for i in l]

def build_network(layer_config, input_tensor, previous_w, input_num, y):
    layer = layer_config['layers']

    network = tf.transpose(input_tensor, [0, 2, 3, 1])
    network = network / network[:, :, -1, 0, None, None]
    network = tf.layers.conv2d(network, 
                               filters = int(layer[0]["filter_number"]),
                               kernel_size = allint(layer[0]["filter_shape"]),
                               strides = allint(layer[0]["strides"]),
                               padding = layer[0]["padding"],
                               activation = layer[0]["activation_function"],
                               kernel_regularizer = layer[0]["regularizer"],
                               # weight_decay=layer[0]["weight_decay"],
                               )
    
    width = network.get_shape()[2]
    network = tf.layers.conv2d(network, 
                               filters = int(layer[1]["filter_number"]),
                               kernel_size = [1, width],
                               strides = [1, 1],
                               padding = "valid",
                               activation = layer[1]["activation_function"],
                               # kernel_regularizer=layer[1]["regularizer"],
                               kernel_regularizer='l2',
                               # weight_decay=layer[1]["weight_decay"],
                               )
    
    width = network.get_shape()[2]
    height = network.get_shape()[1]
    features = network.get_shape()[3]
    network = tf.reshape(network, [input_num, int(height), 1, int(width*features)])
    w = tf.reshape(previous_w, [-1, int(height), 1, 1])
    network = tf.concat([network, w], axis=3)
    network = tf.layers.conv2d(network, 
                               filters = 1, 
                               kernel_size = [1, 1], 
                               padding="valid",
                               # kernel_regularizer = layer[2]["regularizer"],
                               kernel_regularizer = 'l2'
                               # weight_decay=layer[2]["weight_decay"],
                               )
    network = network[:, :, 0, 0]
    btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32, initializer=tf.zeros_initializer)
    btc_bias = tf.tile(btc_bias, [input_num, 1])
    network = tf.concat([btc_bias, network], 1)
    voting = network
    output = tf.nn.softmax(network)
    return output


output = build_network(config_2, input_tensor, previous_w, input_num, y)
future_price = tf.concat([tf.ones([input_num, 1]), y[:, 0, :]], 1)
future_omega = (future_price * output)/tf.reduce_sum(future_price * output, axis=1)[:, None]

commission_ratio = config_2["trading"]["trading_consumption"]

w_t = future_omega[:input_num-1]  # rebalanced
w_t1 = output[1:input_num]
mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*commission_ratio

pv_vector = tf.reduce_sum(output * future_price, reduction_indices=[1])*(tf.concat([tf.ones(1), mu], axis=0)) # 每期的return
pv_value = []
log_mean_free = tf.reduce_mean(tf.log(tf.reduce_sum(output * future_price, reduction_indices=[1])))
portfolio_value = tf.reduce_prod(pv_vector) # return 乘起來就是最後投組價值
mean = tf.reduce_mean(pv_vector)
log_mean = tf.reduce_mean(tf.log(pv_vector))
standard_deviation = tf.sqrt(tf.reduce_mean((pv_vector - mean) ** 2))

sharp_ratio = (mean - 1) / standard_deviation
loss = -tf.reduce_mean(tf.log(pv_vector))


global_step = tf.Variable(0, trainable=False)
learning_rate = config_2["training"]["learning_rate"]
decay_steps = config_2["training"]["decay_steps"]
decay_rate = config_2["training"]["decay_rate"]
learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

#%%
sess.run(tf.global_variables_initializer())

#%% training & logging & testing network

import numpy as np

import time

test_set = data.get_test_set()
train_set = data.get_training_set()
array = np.maximum.reduce(test_set['y'][:, 0, :], 1)
total = 1.0
for i in array:
    total = total * i

# logging.info("upper bound in test is %s" % total)
print("upper bound in test is %s" % total)
            
# init_tensor_board(log_file_dir) # tensorboard related func. omit now
starttime = time.time()

total_data_time = 0
total_training_time = 0
best_metric = 0

por_val = []
tpor_val = []

for i in range(config_2['training']["steps"]):  # train 80000次
    step_start = time.time()

    batch = data.next_batch()
    batch_x = batch["X"]
    batch_y = batch["y"]
    batch_last_w = batch["last_w"]
    batch_setw = batch["setw"]

    finish_data = time.time()
    total_data_time += (finish_data - step_start)
    # self._agent.train(x, y, last_w=last_w, setw=setw)
    
    tflearn.is_training(True, sess)
    results = sess.run([portfolio_value, train_step, output], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})
    batch_setw(results[-1][:, 1:])
    
    total_training_time += time.time() - finish_data
    if i % 1000 == 0:
        print('hi~', i)
        print("average time for data accessing is %s"%(total_data_time/1000))
        print("average time for training is %s"%(total_training_time/1000))
        total_training_time = 0
        total_data_time = 0
        # self.log_between_steps(i)
        tflearn.is_training(False, sess)
        
        batch_x = test_set["X"]
        batch_y = test_set["y"]
        batch_last_w = test_set["last_w"]
        batch_setw = test_set["setw"]
        
        v_pv, v_log_mean, v_loss, v_log_mean_free, weights, sharpe_ratio, pv_vec = \
        sess.run([portfolio_value, log_mean, loss, log_mean_free, output, sharp_ratio, pv_vector], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})
        batch_setw(weights[:, 1:])
        pv = port_value(pv_vec)
        mdd = max_drawdown(pv)
        print('='*30)
        print('step %d' % i)
        print('-'*30)
        
        batch_x = train_set["X"]
        batch_y = train_set["y"]
        batch_last_w = train_set["last_w"]
        batch_setw = train_set["setw"]
        tv_pv, tv_log_mean, tv_loss, tv_log_mean_free, tweights, tsharpe_ratio, tpv_vec = \
        sess.run([portfolio_value, log_mean, loss, log_mean_free, output, sharp_ratio, pv_vector], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})
        batch_setw(tweights[:, 1:])
        tpv = port_value(tpv_vec)
        tmdd = max_drawdown(tpv)
        print('the portfolio value on test set is %s\nlog_mean is %s\n'
                      'loss_value is %3f\nlog mean without commission fee is %3f\nSharpe ratio is %s\nMDD is %s' % \
                      (v_pv, v_log_mean, v_loss, v_log_mean_free, sharpe_ratio, mdd))
        print('-'*30)
        print('the portfolio value on train set is %s\nlog_mean is %s\n'
                     'loss_value is %3f\nlog mean without commission fee is %3f\nSharpe ratio is %s\nMDD is %s' % \
                     (tv_pv, tv_log_mean, tv_loss, tv_log_mean_free, tsharpe_ratio, tmdd))
        por_val.append(v_pv)
        tpor_val.append(tv_pv)
        print('='*30+"\n")
        
        # if v_pv > best_metric:
        #     best_metric = v_pv
        #     print("get better model at %s steps,"
        #                  " whose test portfolio value is %s" % (i, v_pv))

batch_x = test_set["X"]
batch_y = test_set["y"]
batch_last_w = test_set["last_w"]
batch_setw = test_set["setw"]

pv, v_log_mean, sharpe_ratio, pv_vec = sess.run([portfolio_value, log_mean, sharp_ratio, pv_vector], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})
pv_ = port_value(pv_vec)
mdd = max_drawdown(pv_)
print('the portfolio value train No.%s is %s log_mean is %s, Sharpe ratio is %s, MDD is %s,'
                ' the training time is %d seconds' % ('2', pv, v_log_mean, sharpe_ratio, mdd, time.time() - starttime))

batch_x = train_set["X"]
batch_y = train_set["y"]
batch_last_w = train_set["last_w"]
batch_setw = train_set["setw"]

tpv, tlog_mean, tsharpe_ratio, tpv_vec = sess.run([portfolio_value, log_mean, sharp_ratio, pv_vector], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})
tpv_ = port_value(tpv_vec)
tmdd = max_drawdown(tpv_)
print('the portfolio value train No.%s is %s log_mean is %s,Sharpe ratio is %s, MDD is %s,'
                ' the training time is %d seconds' % ('2', tpv, tlog_mean, tsharpe_ratio, tmdd, time.time() - starttime))

