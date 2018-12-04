import tensorflow as tf

# Assume batch size is 1
def generator(sequence_type):
    if sequence_type == 1:
        for i in range(5):
            yield 10 + i
    elif sequence_type == 2:
        for i in range(5):
            yield (30 + 3 * i, 60 + 2 * i)
    elif sequence_type == 3:
        for i in range(1, 4):
            yield (i, ['Hi'] * i)


dataset1 = tf.data.Dataset.from_generator(generator, tf.int32, args = ([1]))
# Emits data of 10, 11, 12, 13, 14 (One element at a time)

dataset2 = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32), args = ([2]))
# Emits data of (30, 60), (33, 62), (36, 64), (39, 66), (42, 68)
# Emits one tuple at a time

dataset3 = tf.data.Dataset.from_generator(generator, (tf.int32, tf.string), args = ([3]))
# Emits data of (1, ['Hi']), (2, ['Hi', 'Hi']), (3, ['Hi', 'Hi', 'Hi'])
# Emits one tuple at a time
