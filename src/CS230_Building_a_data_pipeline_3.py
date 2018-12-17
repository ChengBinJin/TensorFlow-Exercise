import tensorflow as tf

dataset = tf.data.TextLineDataset('./data/file.txt')
dataset = dataset.map(lambda token: tf.string_split([token]).values)

def extract_char(token, default_value="<pad_char>"):
    # Split characters
    out = tf.string_split(token, delimiter='')
    # Convert to Dense tensor, filling with default value
    out = tf.sparse_tensor_to_dense(out, default_value=default_value)
    return out

# Dataset yields word and characters
dataset = dataset.map(lambda token: (token, extract_char(token)))

# Creating the padded batch
padded_shapes = (tf.TensorShape([None]),  # padding the words
                 tf.TensorShape([None, None])) # padding the characters for each word
padding_values = ('<pad_word>',  # sentences padded on the right with <pad>
                  '<pad_char>')  # arrays of characters padded on the right with <pad>

dataset = dataset.padded_batch(1, padded_shapes=padded_shapes, padding_values=padding_values)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(1):
        sentences, characters = sess.run(next_element)
        print(sentences[0])
        print(characters[0])
