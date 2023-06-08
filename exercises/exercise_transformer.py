import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

def positional_embedding(positions, d_model, denom=10_000):
    positional_encoding = np.zeros((positions, d_model))

    for pos in range(positions):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (denom ** ((2 * i) / d_model)))
            if (i + 1) < d_model:
                positional_encoding[pos, i + 1] = np.cos(pos / (denom ** ((2 * i) / d_model)))

    return tf.cast(positional_encoding, dtype=tf.float32)

def plot():
    max_len = 2_000
    d_model = 512

    sinusoidal_pe = positional_embedding(max_len, d_model)
    plt.figure(figsize=(10,10))
    plt.pcolormesh(sinusoidal_pe, cmap='viridis')
    plt.xlabel('i')
    plt.xlim((0, d_model))
    plt.ylim((max_len,0))
    plt.ylabel('pos')
    plt.colorbar()
    plt.title('Sinusoidal Positional Encoding')
    plt.show()


def exercise_1():
    def generate_data(n_samples, length):
        # Generate recurrent sequences of integers, with random parameters
        # WARNING: for modest length, already very large numbers (can) appear
        X = []
        Y = []

        for _ in range(n_samples):
            sample = [np.random.randint(1, 4)]
            a = np.random.randint(0, 3)
            b = np.random.randint(0, 3)
            sample.append(a * sample[-1])
            for _ in range(length-1):
                sample.append(a * sample[-1] + b * sample[-2])
            X.append(sample[:-1])
            Y.append(sample[1:])

        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)
        m = min( np.min(X), np.min(Y) )
        return X-m, Y-m

    n_samples  = 10_000
    max_length = 12

    X, Y = generate_data(n_samples, max_length)
    print(f"X.shape={X.shape},MIN={np.min(X)},MAX={np.max(X)}")

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    vocab_size = max(np.max(X),np.max(Y)) + 1
    embed_dim = 512 # Embedding size for each token
    num_heads = 2   # Number of attention heads
    ff_dim    = 32  # Hidden layer size in feed forward network inside transformer

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = TODO
            self.ffn = TODO
            self.layernorm1 = LayerNormalization() #epsilon=1e-6)
            self.layernorm2 = LayerNormalization() #epsilon=1e-6)

        def call(self, x):
            # TODO ln -> att -> add -> ln -> ffn -> add
            return x

    class Transformer(tf.keras.Model):
        def __init__(self, max_length, vocab_size, embed_dim, num_heads, ff_dim, num_blocks):
            super(Transformer, self).__init__()

            self.max_length = max_length
            self.embed_dim  = embed_dim
            self.num_heads  = num_heads
            self.ff_dim     = ff_dim
            self.num_blocks = num_blocks

            self.embedding    = Embedding(vocab_size, embed_dim)
            self.pos_encoding = positional_embedding(max_length, embed_dim)
            self.transformer_blocks = TODO
            self.final_layer = Dense(vocab_size, activation='softmax')

        def call(self, x):
            # TODO embed -> add -> transformer -> final
            return x

    num_blocks = 2
    model = Transformer(max_length=max_length, vocab_size=vocab_size,
                        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_blocks=num_blocks)

    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(dataset, epochs=25)

if __name__ == '__main__':
    exercise_1()