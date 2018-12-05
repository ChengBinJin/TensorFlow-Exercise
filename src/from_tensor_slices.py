import numpy as np
import tensorflow as tf

# Assume batch size is 1
dataset1 = tf.data.Dataset.from_tensor_slices(tf.range(10, 15))
# Emits data of 10, 11, 12, 13, 14, (One element at a time)

dataset2 = tf.data.Dataset.from_tensor_slices((tf.range(30, 45, 3), np.arange(60, 70, 2)))
# Emits data of (30, 60), (33, 62), (36, 64), (39, 66), (42, 68)
# Emits one tuple at a time

dataset3 = tf.data.Dataset.from_tensor_slices((tf.range(10), np.arange(5)))
# Dataset not possible as zeroth dimension is different at 10 and 5
