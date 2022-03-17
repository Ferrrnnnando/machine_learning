import time
from tqdm import tqdm_notebook  # Show progress bar
from keras.datasets import cifar10
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)    # (50000, 32, 32, 3)

# Normalize Data


def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

# Shuffle training data for each epoch


def Shuffle_data(x_train, y_train):
    shuffle = list(zip(x_train, y_train))
    np.random.shuffle(shuffle)

    x_train, y_train = zip(*shuffle)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return (x_train, y_train)

# Create Batch data


def Train_in_batch(x_train, y_train, batch_size):
    assert len(x_train) / batch_size == int(len(x_train) / batch_size)

    batch_x = np.split(x_train, len(x_train) / batch_size)
    batch_y = np.split(y_train, len(y_train) / batch_size)

    for x, y in zip(batch_x, batch_y):
        yield (x, y)


# Step 1: Normalize training and test data
x_train, x_test = normalize(x_train, x_test)

# Onehot encoding the labels: y_train and y_test
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train).toarray()
y_test = one_hot.fit_transform(y_test).toarray()

tf.reset_default_graph()

#############################
# CNN starts here


def Convolution_Block(input_, filters=16, strides=(1, 1), kernel_size=(3, 3), padding='same'):
    # Convolution Layer
    X = tf.layers.conv2d(inputs=input_, filters=filters,
                         kernel_size=kernel_size, strides=strides, padding=padding)

    # Activation function
    X = tf.nn.leaky_relu(X)

    # Batch Normalization
    output = tf.layers.batch_normalization(X)
    return output


with tf.name_scope('input'):
    # 4-D input: [Batch_size,height,width,channels]
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    # One hot label
    y_true = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('stem'):
    # 1st conv layer
    X = Convolution_Block(inputs, filters=16, strides=(
        1, 1), kernel_size=(3, 3), padding='same')
    print(X.shape)
    # 1st max pooling layer
    X = tf.nn.max_pool(X, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    print(X.shape)

    # 2nd conv layer
    X = Convolution_Block(X, filters=16, strides=(
        1, 1), kernel_size=(3, 3), padding='valid')
    print(X.shape)
    # 2nd max pooling layer
    X = tf.nn.max_pool(X, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    print(X.shape)

    # 3rd conv layer
    X = Convolution_Block(X, filters=8, strides=(
        1, 1), kernel_size=(3, 3), padding='same')
    print(X.shape)

with tf.name_scope('Flatten'):
    X = tf.layers.Flatten()(X)
    print(X.shape)

with tf.name_scope('FCL'):
    X = tf.layers.dense(X, 100, name='dense_1')
    X = tf.nn.leaky_relu(X)
    X = tf.layers.dropout(X, rate=0.3)
    out_put_final = tf.layers.dense(X, 10, name='out_put',)
    print(out_put_final.shape)

with tf.name_scope('Predict'):
    # pass output to softmax func
    prediction = tf.nn.softmax(out_put_final)

with tf.name_scope('loss'):
    # Use tf.reduce_mean to average batch's losses
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true, logits=out_put_final, name='loss_'))
    # Target is to use minimize loss, and we use Adam optimizer
    optim = tf.train.AdamOptimizer().minimize(loss)


epochs = 50
batch_size = 100
iteration = int(len(x_train)/batch_size)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print('Size of x_test: ', x_test.nbytes/1024/1024, 'Mb')

    for epoch in range(epochs):

        x_train, y_train = Shuffle_data(x_train, y_train)
        Batch_data = Train_in_batch(x_train, y_train, batch_size=batch_size)

        training_loss = 0
        training_acc = 0
        bar = tqdm_notebook(range(iteration))  # Create a progress bar
        which_pic = 0
        for iter_ in bar:
            train_batch_x, train_batch_y = next(Batch_data)
            # 更新weights，以及得到prediction
            tr_pred, training_loss_batch, _ = sess.run([prediction, loss, optim], feed_dict={
                                                       inputs: train_batch_x, y_true: train_batch_y, })
            training_loss += training_loss_batch

            training_acc_batch = accuracy_score(
                np.argmax(train_batch_y, axis=1), np.argmax(tr_pred, axis=1))
            training_acc += training_acc_batch
            if iter_ % 5 == 0:
                bar.set_description('loss: %.4g' % training_loss_batch)
                # 每5次batch更新顯示的batch loss(進度條前面)'''

        training_loss /= iteration
        training_acc /= iteration

        te_pred, testing_loss = sess.run([prediction, loss], feed_dict={
                                         inputs: x_test, y_true: y_test})

        #### calculate testing data acc ####
        testing_acc = accuracy_score(
            np.argmax(y_test, axis=1), np.argmax(te_pred, axis=1))
        print('Training_set accuracy: ', training_acc)
        print('Test_set accuracy: ', testing_acc)
