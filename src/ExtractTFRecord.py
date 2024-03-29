import tensorflow as tf

def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'int_list1': tf.FixedLenFeature([], tf.int64),
        'float_list1': tf.FixedLenFeature([], tf.float32),
        'str_list1': tf.FixedLenFeature([], tf.string),
        # If size is different of different records, use VarLenFeature
        'float_list2': tf.VarLenFeature(tf.float32)
    }
    sample = tf.parse_single_example(data_record, features)
    return sample

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['./data/example.tfrecord'])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            data_record = sess.run(next_element)
            print(data_record)
    except:
        pass
