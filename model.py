import random
from time import sleep

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class Net(object):
    def __init__(self):
        self.batch_size = 64
        self.lr = 0.001
        self.x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
        self.y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes
        layer = tf.layers.dense(self.x, 128, activation=tf.nn.relu)
        self.logits = tf.layers.dense(layer, 10)
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self):
        batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
        fd = {
            self.x: batch_xs,
            self.y: batch_ys
        }
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict=fd)
        print(loss)

    def close(self):
        self.session.close()

    def set_weights(self, name, data):
        pass

    def weights_iter(self):
        return []


class FakeNet(Net):
    def __init__(self):
        self.d = {
            'a': b'a',
            'b': b'b',
            'c': b'c',
        }

    def train(self):
        for name in self.d:
            self.d[name] += b'.'
        t = random.randint(1, 3)
        sleep(t)

    def set_weights(self, name, data):
        self.d[name] = data

    def weights_iter(self):
        for name, data in self.d.items():
            yield name, data
