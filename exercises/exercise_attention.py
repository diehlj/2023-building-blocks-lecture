
def ex_softmax():
    # Implement the softmax function from scratch
    # and compare with the results of tf.nn.softmax.
    pass

def ex_attention():
    # Implement the Attention layer from scratch.
    pass

    # Implement the MultiHeadAttention layer from scratch.
    pass

def ex_counting():
    # Use one (multihead) attention layer (and possible FFNs)
    # to learn to count the number of times
    # the last letter in a random string appears in the string
    # 'abbca' -> 2 
    # 'abbcb' -> 3
    # 'bbbba' -> 1
    pass

def ex_gender():
    import tensorflow as tf
    import numpy as np
    import os
    
    # raw_data = np.loadtxt(os.path.dirname(os.path.abspath(__file__))+'/gender_data_50_510.txt', dtype=str, delimiter=';', converters=lambda s: s.strip())
    raw_data = np.loadtxt(os.path.dirname(os.path.abspath(__file__))+'/gender_data_25_500.txt', dtype=str, delimiter=';', converters=lambda s: s.strip())
    raw_data_training = raw_data[0:int(0.8*len(raw_data))]
    raw_data_test     = raw_data[int(0.8*len(raw_data)):]
    max_features = 10_000  # Maximum vocab size.
    max_len = 8            # Sequence length to pad the outputs to.
    # TODO Use TextVectorization to turn the strings into sequences of numbers.
    vectorize_input_layer = TODO
    vectorize_input_layer.adapt(raw_data[:,0])
    print( vectorize_input_layer(raw_data[0:1,0] ))

    # For the output it is easier:
    vectorize_output_layer = tf.keras.layers.StringLookup(vocabulary=['der','die','das'])
    print( vectorize_output_layer(raw_data[:,1]) )

    class MyModel(tf.keras.Model):
        # TODO Implement: vectorize -> embedding -> attention -> dense -> dense
    model = MyModel(max_features, vectorize_input_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # TODO Pick a loss:
    loss = TODO
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(raw_data_training[:,0],  vectorize_output_layer(raw_data_training[:,1]), epochs=100, batch_size=8)
    model.evaluate(raw_data_test[:,0], vectorize_output_layer(raw_data_test[:,1]), verbose=2)

    # TODO If learning works: investigate the attention weights.

    # Use one self-attention layer (and possible FFNs)
    # to learn to differentiate between
    pass