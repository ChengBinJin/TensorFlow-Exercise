import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

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


def map_fn(x, y):
    # Do transformations here
    return x, y

epochs = 1
batch_size = 64

placeholder_X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
placeholder_y = tf.placeholder(tf.float32, shape=[None])

# Create separate Datasets for training, validation and testing
train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: map_fn(x, y))

val_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
val_dataset = val_dataset.batch(batch_size)

y_test = np.array(y_test, dtype=np.int32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# Feedable iterator assigns each iterator a unique string handle it is going to work on
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
data_X, data_y = iterator.get_next()
data_y = tf.cast(data_y, tf.int32)
model = Model(data_X, data_y)

# Create Reinitializable iterator for Train and Validation, one shot iterator for Test
train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_iterator = train_val_iterator.make_initializer(train_dataset)
val_iterator = train_val_iterator.make_initializer(val_dataset)
test_iterator = test_dataset.make_one_shot_iterator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Create string handles for above reinitializable and one shot iterators.
    train_val_string = sess.run(train_val_iterator.string_handle())
    test_string = sess.run(test_iterator.string_handle())

    for epoch_no in range(epochs):
        train_loss, train_accuracy = 0., 0.
        val_loss, val_accuracy = 0., 0.

        # Start reinitializable's train iterator
        sess.run(train_iterator, feed_dict={placeholder_X: X_train, placeholder_y: y_train})
        try:
            with tqdm(total=len(y_train)) as pbar:
                while True:
                    # Feed to feedable iterator the sring handle of reinitialize iterator
                    _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy],
                                            feed_dict={handle: train_val_string})

                    train_loss += loss
                    train_accuracy += acc
                    pbar.update(batch_size)
        except tf.errors.OutOfRangeError:
            pass

        # Start reinitializable's validation iterator
        sess.run(val_iterator, feed_dict={placeholder_X: X_val, placeholder_y: y_val})
        try:
            while True:
                loss, acc = sess.run([model.loss, model.accuracy], feed_dict={handle: train_val_string})
                val_loss += loss
                val_accuracy += acc
        except tf.errors.OutOfRangeError:
            pass

        print('\nEpoch: {}'.format(epoch_no + 1))
        print('Training accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / len(y_train),
                                                               train_loss / len(y_train)))
        print('val accuracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / len(y_val), val_loss / len(y_val)))

    # test_loss, test_accuracy = 0., 0.
    # try:
    #     while True:
    #         # Feed to iterator the string handler of one shot iterator
    #         loss, acc = sess.run([model.loss, model.accuracy], feed_dict={handle: test_string})
    #         test_loss += loss
    #         test_accuracy += acc
    # except tf.errors.OutOfRangeError:
    #     pass
    #
    # print('\n Test accuracy: {:.4f}, loss: {:.4f}'.format(test_accuracy / len(y_test), test_loss / len(y_test)))




