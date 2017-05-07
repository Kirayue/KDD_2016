from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import produce_testdata as pt 
import random
import sys
# Parameters
learning_rate = 0.001
training_iters = int(sys.argv[3])
batch_size = 30
display_step = 10
#model_path = "./model/1D_{}_model_{}_{}.ckpt".format(sys.argv[3], sys.argv[1], sys.argv[2])
model_path = "./model/{}_model_{}_{}.ckpt".format(sys.argv[3], sys.argv[1], sys.argv[2])

# Network Parameters
#n_input = 1 
n_input = 22
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
    


# ReadData
#datapath = './data/{}_{}_1D_'.format(sys.argv[1], sys.argv[2])
datapath = './data/{}_{}_'.format(sys.argv[1], sys.argv[2])
traindata, trainanswer = readData(datapath + 'train_data.npy', datapath + 'train_data_ans.npy') 
validdata, validanswer = readData(datapath + 'valid_data.npy', datapath + 'valid_data_ans.npy')

#with open('./temp/1D_{}_{}_trainrecord'.format(sys.argv[1], sys.argv[2]), 'r') as f:
with open('./temp/{}_{}_trainrecord'.format(sys.argv[1], sys.argv[2]), 'r') as f:
    trainformat = f.read()
trainformat = trainformat.split('\n')
#with open('./temp/1D_{}_{}_validrecord'.format(sys.argv[1], sys.argv[2]), 'r') as f:
with open('./temp/{}_{}_validrecord'.format(sys.argv[1], sys.argv[2]), 'r') as f:
    validformat = f.read()
validformat = validformat.split('\n')

# tf Graph input
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

merged = tf.summary.merge_all()
# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #file_writer = tf.summary.FileWriter('./logs/1D_{}_{}_log'.format(sys.argv[1], sys.argv[2]), sess.graph)
    file_writer = tf.summary.FileWriter('./logs/{}_{}_log'.format(sys.argv[1], sys.argv[2]), sess.graph)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = getBatch(batch_size, traindata, trainanswer)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            MAPE = sess.run(evaluation, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            summary= sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            file_writer.add_summary(summary, step)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", MAPE= " + \
                  "{:.5f}".format(MAPE))
        step += 1
    print("Optimization Finished!")

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: {}".format(save_path))

    # Add prediction traindata to trainformat
    trainpred = sess.run(pred, feed_dict={x:traindata})

    for index, value in enumerate(trainformat):
        trainformat[index] += ",{}".format(trainpred[index][0])
        #print(trainformat[index])

    validdata, validanswer = getBatch(70, validdata, validanswer)
    validpred = sess.run(pred, feed_dict={x:validdata})
    for index, value in enumerate(validformat):
        validformat[index] += ",{}".format(validpred[index][0])
        #print(trainformat[index])
    #print(trainformat[0])
    #with open('./temp/1D_{}_{}_{}_trainpred'.format(sys.argv[3], sys.argv[1], sys.argv[2]), 'w') as f:
    with open('./temp/{}_{}_{}_trainpred'.format(sys.argv[3], sys.argv[1], sys.argv[2]), 'w') as f:
        f.write('\n'.join(trainformat))    

    #with open('./temp/1D_{}_{}_{}_validpred'.format(sys.argv[3], sys.argv[1], sys.argv[2]), 'w') as f:
    with open('./temp/{}_{}_{}_validpred'.format(sys.argv[3], sys.argv[1], sys.argv[2]), 'w') as f:
        f.write('\n'.join(validformat))    
    # Validation
    print("Valid MAPE:", \
        sess.run(evaluation, feed_dict={x: validdata, y: validanswer}))
