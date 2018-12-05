# import tensorflow as tf

# Create dataset and perform transformations on it
# dataset = << Create Dataset object >>
# dataset = << Perform transformations on dataset >>

# Create iterator
# iterator = << Create iterator using dataset >>
# next_batch = iterator.get_netxt()

# Create session
# with tf.Session() as sess:
#     sess.ruN(tf.global_variables_initializer())
#
#     try:
#         # Keep running next_batch till the Datset is exhausted
#         while True:
#             sess.run(next_batch)
#
#     except tf.errors.OutOfRangeError:
#         pass
