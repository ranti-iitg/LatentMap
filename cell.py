import tensorflow as tf


class ConvGRUCell(tf.contrib.rnn.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, height, width, filters, kernel, initializer=None, activation=tf.tanh):
    self._height = height
    self._width = width
    self._filters = filters
    self._kernel = kernel
    self._initializer = initializer
    self._activation = activation
    self._size = int(self._height * self._width * self._filters)

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, input, state, scope=None):
    with tf.variable_scope(scope or self.__class__.__name__):

      with tf.variable_scope('Expand'):
        samples = input.get_shape()[0].value
        shape = [samples, self._height, self._width, -1]
        input = tf.reshape(input, shape)
        state = tf.reshape(state, shape)

      with tf.variable_scope('Gates'):
        channels = input.get_shape()[-1].value
        x = tf.concat([input, state], axis=3)
        n = channels + self._filters
        m = 2 * self._filters if self._filters > 1 else 2
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(x, W, 'SAME')
        y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(1.0))
        reset_gate, update_gate = tf.split(y, 2, axis=3)
        reset_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(update_gate)

      with tf.variable_scope('Output'):
        x = tf.concat([input, reset_gate * state], axis=3)
        n = channels + self._filters
        m = self._filters
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(x, W, 'SAME')
        y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(0.0))
        y = self._activation(y)
        output = update_gate * state + (1 - update_gate) * y

      with tf.variable_scope('Flatten'):
        output = tf.reshape(output, [-1, self._size])

      return output, output


def flatten(tensor):
  samples, timesteps, height, width, filters = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height * width * filters])


def expand(tensor, height, width, filters):
  samples, timesteps, features = tensor.get_shape().as_list()
  return tf.reshape(tensor, [samples, timesteps, height, width, filters])
