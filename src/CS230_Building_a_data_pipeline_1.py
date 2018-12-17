import tensorflow as tf

dataset = tf.data.TextLineDataset('data/file.txt')
dataset = dataset.map(lambda string: tf.string_split([string]).values)
dataset = dataset.shuffle(buffer_size=3)
dataset = dataset.batch(2)
dataset = dataset.prefetch(1)

# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess:
    # Initialize the iterator
    sess.run(init_op)
    print(sess.run(next_element))
    print(sess.run(next_element))
    # Move the iterator back to the beginning
    sess.run(init_op)
    print(sess.run(next_element))
