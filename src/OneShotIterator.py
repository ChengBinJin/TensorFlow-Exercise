import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim

# LeNet-5 model
class Model:
    def __init__(self, data_X, data_y):
        self.n_class = 10
        self._create_architecture(data_X, data_y)

    def _create_architecture(self, data_X, data_y):
        y_hot = tf.one_hot(data_y, depth=self.n_class)
        logits = self._create_model(data_X)
        predictions = tf.argmax(logits, 1, output_type=tf.int32)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_hot, logits=logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, data_y), tf.float32))

    def _create_model(self, X):
        X1 = X - 0.5
        X1 = tf.pad(X1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1)):
            net = slim.conv2d(X1, 6, [5, 5], padding='VALID')
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 16, [5, 5], padding='VALID')
            net = slim.max_pool2d(net, [2, 2])

            net = tf.reshape(net, [-1, 400])
            net = slim.fully_connected(net, 120)
            net = slim.fully_connected(net, 84)
            net = slim.fully_connected(net, self.n_class, activation_fn=None)
        return net

# Extractomg MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_val, y_val = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels


epochs = 10
batch_size = 64
iterations = len(y_train) * epochs

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# Generate the complete Dataset required in the pipeline
dataset = dataset.repeat(epochs).batch(batch_size)
iterator = dataset.make_one_shot_iterator()

data_X, data_y = iterator.get_next()
data_y = tf.cast(data_y, tf.int32)
model = Model(data_X, data_y)

# with tf.Session() as sess, tqdm(total=iterations) as pbar:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tot_accuracy = 0
    try:
        while True:
            accuracy, _ = sess.run([model.accuracy, model.optimizer])
            tot_accuracy += accuracy
            # pbar.update(batch_size)
    except tf.errors.OutOfRangeError:
        pass

print('\nAvearge trainig accuracy: {:.4f}'.format(tot_accuracy / iterations))
