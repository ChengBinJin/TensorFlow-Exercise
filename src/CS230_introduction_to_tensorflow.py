import tensorflow as tf

x = tf.constant(1., dtype=tf.float32, name='my-node-x')
print(x)

with tf.variable_scope('model'):
    x1 = tf.get_variable('x', [], dtype=tf.float32)  # get or crate variable with name 'model/x:0'
    print(x1)

with tf.variable_scope('model', reuse=True):
    x2 = tf.get_variable('x', [], dtype=tf.float32)
    print(x2)

with tf.Session() as sess:
    print(sess.run(x))

    sess.run(tf.global_variables_initializer())  # Initialize the Variables
    sess.run(tf.assign(x1, tf.constant(1.)))  # Change the value of x1
    sess.run(tf.assign(x2, tf.constant(2.)))  # Change the value of x2
    print("x1 = ", sess.run(x1), " x2 = ", sess.run(x2))
