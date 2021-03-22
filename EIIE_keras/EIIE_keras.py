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

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
np.random.seed(0)
tf.random.set_random_seed(0)
sess = tf.Session(config=tf_config)
# %% load config & data
config_2 = load_config()
data = DataMatrices.create_from_config(config_2)
# %% network structure
feature_number = config_2["input"]["feature_number"]
rows = config_2["input"]["coin_number"]
columns = config_2["input"]["window_size"]
input_tensor = tf.placeholder(tf.float32, shape=[None, 5, 5, 3])  # [None, features, rows, columns]
previous_w = tf.placeholder(tf.float32, shape=[None, 5])  # [None, rows]
input_num = tf.placeholder(tf.int32, shape=[])  # [rows]
y = tf.placeholder(tf.float32, shape=[None, 5, 5])  # [None,feature_number,rows]


def allint(l):
    return [int(i) for i in l]


def build_network(layer_config, input_tensor, previous_w, input_num, y):
    network = tf.transpose(input_tensor, [0, 2, 3, 1])
    network = network / network[:, :, -1, 0, None, None]
    network = tf.layers.conv2d(network,
                               # network[0]
                               filters=5,  # ["filter_number"]
                               kernel_size=[1, 3],  # ["filter_shape"]
                               strides=[1, 1],
                               padding='valid',
                               activation="relu",
                               # kernel_regularizer = "null",
                               # weight_decay=layer[0]["weight_decay"],
                               )

    width = network.get_shape()[2]
    network = tf.layers.conv2d(network,
                               # network[1]
                               filters=10,
                               kernel_size=[1, width],
                               strides=[1, 1],
                               padding="valid",
                               activation='relu',
                               # kernel_regularizer=layer[1]["regularizer"],
                               kernel_regularizer='l2',
                               # weight_decay=layer[1]["weight_decay"],
                               )

    width = network.get_shape()[2]
    height = network.get_shape()[1]
    features = network.get_shape()[3]
    network = tf.reshape(network, [input_num, int(height), 1, int(width * features)])
    w = tf.reshape(previous_w, [-1, int(height), 1, 1])
    network = tf.concat([network, w], axis=3)
    network = tf.layers.conv2d(network,
                               filters=1,
                               kernel_size=[1, 1],
                               padding="valid",
                               # kernel_regularizer = layer[2]["regularizer"],
                               kernel_regularizer='l2'
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
future_omega = (future_price * output) / tf.reduce_sum(future_price * output, axis=1)[:, None]
commission_ratio = config_2["trading"]["trading_consumption"]
w_t = future_omega[:input_num - 1]  # rebalanced
w_t1 = output[1:input_num]
mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:] - w_t[:, 1:]), axis=1) * commission_ratio
pv_vector = tf.reduce_sum(output * future_price, reduction_indices=[1]) * (tf.concat([tf.ones(1), mu], axis=0))

log_mean_free = tf.reduce_mean(tf.log(tf.reduce_sum(output * future_price, reduction_indices=[1])))
portfolio_value = tf.reduce_prod(pv_vector)
mean = tf.reduce_mean(pv_vector)
log_mean = tf.reduce_mean(tf.log(pv_vector))
standard_deviation = tf.sqrt(tf.reduce_mean((pv_vector - mean) ** 2))
sharp_ratio = (mean - 1) / standard_deviation
# loss = -tf.reduce_mean(tf.log(tf.cond(tf.equal(pv_vector, 0), true_fn=lambda: 0, false_fn=lambda: -pv_vector)))
loss = -tf.reduce_mean(tf.log(pv_vector))
# regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# if regularization_losses:
#     for regularization_loss in regularization_losses:
#         loss += regularization_loss
global_step = tf.Variable(0, trainable=False)
learning_rate = config_2["training"]["learning_rate"]
decay_steps = config_2["training"]["decay_steps"]
decay_rate = config_2["training"]["decay_rate"]
learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
# %%
# tf.reset_default_graph()
sess.run(tf.global_variables_initializer())
# %% training & logging & testing network
import numpy as np
import logging
import time

# train_dir = "train_package"
# s_path = "./" + train_dir + "/" + '2'+ "/netfile"
# l_path = "./" + train_dir + "/" + '2' + "/tensorboard"
# console_level = logging.INFO
# logfile_level = logging.DEBUG
# logging.basicConfig(filename=l_path.replace("tensorboard","programlog"), level=console_level)
# console = logging.StreamHandler()
# console.setLevel(console_level)
# logging.getLogger().addHandler(console)
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
for i in range(config_2['training']["steps"]):
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

    results = sess.run([train_step, output], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w,
                                                        input_num: batch_x.shape[0]})
    batch_setw(results[-1][:, 1:])

    total_training_time += time.time() - finish_data
    if i % 1000 == 0:
        print('hi~', i)
        # logging.info("average time for data accessing is %s"%(total_data_time/1000))
        # logging.info("average time for training is %s"%(total_training_time/1000))
        print("average time for data accessing is %s" % (total_data_time / 1000))
        print("average time for training is %s" % (total_training_time / 1000))
        total_training_time = 0
        total_data_time = 0
        # self.log_between_steps(i)
        tflearn.is_training(False, sess)

        batch_x = test_set["X"]
        batch_y = test_set["y"]
        batch_last_w = test_set["last_w"]
        batch_setw = test_set["setw"]

        v_pv, v_log_mean, v_loss, v_log_mean_free, weights = \
            sess.run([portfolio_value, log_mean, loss, log_mean_free, output],
                     feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w,
                                input_num: batch_x.shape[0]})

        batch_setw(weights[:, 1:])

        # logging.info('='*30)
        # logging.info('step %d' % i)
        # logging.info('-'*30)
        print('=' * 30)
        print('step %d' % i)
        print('-' * 30)

        batch_x = train_set["X"]
        batch_y = train_set["y"]
        batch_last_w = train_set["last_w"]
        batch_setw = train_set["setw"]

        # loss_value = sess.run([loss], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})

        # logging.info('training loss is %s\n' % loss_value)
        # logging.info('the portfolio value on test set is %s\nlog_mean is %s\n'
        #              'loss_value is %3f\nlog mean without commission fee is %3f\n' % \
        #              (v_pv, v_log_mean, v_loss, v_log_mean_free))
        # logging.info('='*30+"\n")

        # print('training loss is %s\n' % loss_value)
        print('the portfolio value on test set is %s\nlog_mean is %s\n'
              'loss_value is %3f\nlog mean without commission fee is %3f\n' % \
              (v_pv, v_log_mean, v_loss, v_log_mean_free))
        print('=' * 30 + "\n")

        if v_pv > best_metric:
            best_metric = v_pv
            # logging.info("get better model at %s steps,"
            #              " whose test portfolio value is %s" % (i, v_pv))
            print("get better model at %s steps,"
                  " whose test portfolio value is %s" % (i, v_pv))
batch_x = test_set["X"]
batch_y = test_set["y"]
batch_last_w = test_set["last_w"]
batch_setw = test_set["setw"]
pv, log_mean = sess.run([portfolio_value, log_mean],
                        feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w,
                                   input_num: batch_x.shape[0]})
# logging.warning('the portfolio value train No.%s is %s log_mean is %s,'
#                 ' the training time is %d seconds' % ('2', pv, log_mean, time.time() - starttime))
print('the portfolio value train No.%s is %s log_mean is %s,'
      ' the training time is %d seconds' % ('2', pv, log_mean, time.time() - starttime))
# return self.__log_result_csv(index, time.time() - starttime)
# %% back_test
# from pgportfolio.trade import backtest
# dataframe = None
# csv_dir = './train_package/train_summary.csv'
# tflearn.is_training(False, sess)
# v_pv, v_log_mean, v_loss, v_log_mean_free, weights = \
# sess.run([portfolio_value, log_mean, loss, log_mean_free, output], feed_dict={input_tensor: batch_x, y: batch_y, previous_w: batch_last_w, input_num: batch_x.shape[0]})
# batch_setw(weights[:, 1:])
# backtest = backtest.BackTest(config_2.copy(),
#                              net_dir=None,
#                              agent=self._agent)
# backtest.start_trading()
# result = Result(test_pv=[v_pv],
#                 test_log_mean=[v_log_mean],
#                 test_log_mean_free=[v_log_mean_free],
#                 test_history=[''.join(str(e)+', ' for e in benefit_array)],
#                 config=[json.dumps(self.config)],
#                 net_dir=[index],
#                 backtest_test_pv=[backtest.test_pv],
#                 backtest_test_history=[''.join(str(e)+', ' for e in backtest.test_pc_vector)],
#                 backtest_test_log_mean=[np.mean(np.log(backtest.test_pc_vector))],
#                 training_time=int(time))
# new_data_frame = pd.DataFrame(result._asdict()).set_index("net_dir")
# if os.path.isfile(csv_dir):
#     dataframe = pd.read_csv(csv_dir).set_index("net_dir")
#     dataframe = dataframe.append(new_data_frame)
# else:
#     dataframe = new_data_frame
# if int(index) > 0:
#     dataframe.to_csv(csv_dir)
# return result
