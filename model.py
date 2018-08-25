import logging
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

logger = logging.getLogger(__name__)


class Net(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 128)
        lr = kwargs.get('lr', 0.001)
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.labels = tf.placeholder(tf.float32, [None, 10])
        layer = tf.layers.dense(self.input, 128, activation=tf.nn.relu)
        self.logits = tf.layers.dense(layer, 10)
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self):
        batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
        fd = {
            self.input: batch_xs,
            self.labels: batch_ys,
        }
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict=fd)
        return loss

    def close(self):
        self.session.close()

    def set_weights(self, name, data):
        logger.debug('set_weights: %s', name)
        var = self.session.graph.get_tensor_by_name(name)

        if not isinstance(data, (list, tuple)):
            data = [data]

        data = [np.frombuffer(d, np.float32) for d in data]  # todo: read type from rpc tensor
        arr = reduce(lambda a, b: a + b, data)
        if len(data) > 1:
            arr = arr / len(data)
        arr = np.reshape(arr, var.shape)
        with self.session.graph.as_default():
            tensor = tf.convert_to_tensor(arr)
            tf.assign(var, tensor)

    def weights_iter(self):
        logger.debug(tf.trainable_variables())
        ops = [n.name for n in self.session.graph.as_graph_def().node if "Variable" in n.op]
        for op_name in ops:
            var_name = op_name + ':0'  # todo: get list of tensors instead of patching op with :X
            data: np.ndarray = self.session.run(var_name)
            logger.debug('weight iter %s', var_name)
            yield var_name, data.tobytes()
