import numpy as np
import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensors(tf.range(10, 15))
# Emits data of [10, 11, 12, 13, 14]

dataset2 = tf.data.Dataset.from_tensors((tf.range(30, 45, 3), np.arange(60, 70, 2)))
# Emits data of ([30, 33, 36, 39, 42], [60, 62, 64, 66, 68])
# Holds entire tuple as one element

dataset3 = tf.data.Dataset.from_tensors((tf.range(10), np.arange(5)))
# Possible with from_tensors, regardless of zeroth dimension mismatch of constituent elements.
# Emits data of ([1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
# Holds entire tuple as one element
