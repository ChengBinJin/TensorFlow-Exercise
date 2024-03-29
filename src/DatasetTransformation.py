import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slice(tf.range(10))
# Create a dataset with data of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

dataset = dataset.repeat(2)
# Duplicate the dataset
# Data will be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

dataset = dataset.shuffle(5)
# Shuffle the dataset
# Assumed shuffling: [3, 0, 7, 9, 4, 2, 5, 0, 1, 7, 5, 9, 4, 6, 2, 8, 6, 8, 1, 3]

def map_fn(x):
    return x * 3


dataset = dataset.map(map_fn)
# Same as dataset = dataset.map(lambda x: x * 3)
# Multiply each element with 3 using map transformation
# Dataset: [9, 0, 21, 27, 12, 6, 15, 0, 3, 21, 15, 27, 12, 18, 6, 24, 18, 24, 3, 9]
