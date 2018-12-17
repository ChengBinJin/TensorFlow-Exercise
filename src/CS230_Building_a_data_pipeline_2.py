import tensorflow as tf

# Load txt file, one example per line
sentences = tf.data.TextLineDataset("./data/sentences.txt")
labels = tf.data.TextLineDataset("./data/labels.txt")

# Zip the sentence and the albels together
dataset = tf.data.Dataset.zip((sentences, labels))

# Create a one shot iterator over the zipped dtaset
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# Actually run in a session
with tf.Session() as sess:
    for i in range(2):
        print(sess.run(next_element))
