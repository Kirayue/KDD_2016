from __future__ import print_function
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from tensorflow.contrib import rnn
import produce_testdata as pt 
import cal_avg_volume as statistic 
import random
import sys

# Parameters
learning_rate = 0.001
training_iters = 40000
batch_size = 30
display_step = 10
global_var = {
    "input": ['/home/kirayue/KDD/data/scripts/training_20min_avg_volume.csv', '/home/kirayue/KDD/data/scripts/test1_20min_avg_volume.csv', '/home/kirayue/KDD/data/dataSets/testing_phase1/weather (table 7)_test1.csv'],
    "stop": '2016-11-18',
    #"model_path": "./model/1D_{}_model_{}_{}.ckpt".format(sys.argv[3], sys.argv[1], sys.argv[2])
    "model_path": "./model/{}_model_{}_{}.ckpt".format(sys.argv[3], sys.argv[1], sys.argv[2])
    }

pred_combine = ['11', '10', '20', '31', '30']
if sys.argv[1] == '0':
    minute = ['00']
else:
    minute = [sys.argv[1]]

if sys.argv[2] == '1':
    hour = ['08', '17']
else:
    hour = ['09', '18']
# hour = ['08', '09', '17', '18']
# minute = ['00', '20', '40']
# Network Parameters
n_input = 22 
#n_input = 1
n_steps = 6 # timesteps
n_hidden = 130 # hidden layer num of features
n_classes = 1 

def readData(filename, answerfile):
    return np.load(filename), np.load(answerfile)

def getBatch(size, data, answer):
    temp = list(zip(data, answer))
    random.shuffle(temp)
    x, y = zip(*temp)
    return np.array(x[:size]), np.reshape(np.array(y[:size]), (size, 1))
    
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#datapath = './data/{}_{}_1D_'.format(sys.argv[1], sys.argv[2])
datapath = './data/{}_{}_'.format(sys.argv[1], sys.argv[2])
validdata, validanswer = readData(datapath + 'valid_data.npy', datapath + 'valid_data_ans.npy')
rawtestdata = pt.readRawData(global_var['input'][1])
rawtraindata = pt.readRawData(global_var['input'][0])
data = pt.preprocessData(rawtraindata + rawtestdata)
avg_volume = statistic.avg_volume(data, global_var['stop'])


with tf.name_scope('input'):
    x = tf.placeholder("float", [None, n_steps, n_input], name = 'x-input')
    y = tf.placeholder("float", [None, n_classes], name = 'y-input')

# Define weights
with tf.name_scope('weights'):
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
with tf.name_scope('biases'):
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

with tf.name_scope('RNN'):
    pred = RNN(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('Error'):
    cost = tf.reduce_mean(tf.pow(y - pred, 2))
    tf.summary.scalar('Cost', cost)
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('MAPE'):
    evaluation = tf.reduce_mean(tf.abs(tf.divide(pred - y, y)))
    tf.summary.scalar('MAPE', evaluation)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, global_var["model_path"])
    print("Model restored from file: {}".format(global_var['model_path']))
    validdata, validanswer = getBatch(70, validdata, validanswer)
    print("Valid MAPE:", \
        sess.run(evaluation, feed_dict={x: validdata, y: validanswer}))

    # Produce Train data and answer
    #sub = ['tollgate_id,time_window,direction,volume']
    sub = []
    for i in range(7): #predition seven days
        for h in hour:  # hour
            for m in minute:  # minute 
                for e in pred_combine:
                    gateid = e[0]
                    direction = e[1]
                    preddate = "{}-{}-2016-10-{} {}:{}:00".format(gateid, direction, i + 18, h, m)
                    testdata = pt.produceData(data, avg_volume, preddate)
                    rowsub = pt.produceSub(preddate)
                    predresult = sess.run(pred, feed_dict={x:testdata})
                    rowsub += "," + str(predresult[0][0])
                    sub.append(rowsub)
                    #data[preddate] = {'volume': predresult[0][0]}
    #prefix = '/home/kirayue/KDD/model/rnn/pred/1D_{}_{}_{}_'.format(sys.argv[3], sys.argv[1], sys.argv[2])
    prefix = '/home/kirayue/KDD/model/rnn/pred/{}_{}_{}_'.format(sys.argv[3], sys.argv[1], sys.argv[2])
    with open(prefix + 'sub_avg.csv', 'w') as f:
        f.write('\n'.join(sub))
print(datapath)
