import tensorflow as tf
layers = tf.keras.layers


class ChunkFeedForward(layers.Layer):
    def __init__(self, hp):
        super(ChunkFeedForward, self).__init__()
        self.n_chunks = hp.model.n_chunks
        self.fc1 = layers.Dense(hp.model.d_ff)
        self.fc2 = layers.Dense(hp.model.d_model)
        self.dropout = layers.Dropout(hp.model.dropout_rate)

    def call(self, x, random=True, padding_mask=0):
        chunks = tf.split(x, num_or_size_splits=self.n_chunks, axis=1)

        xs = [tf.nn.relu(self.fc1(chunk)) for chunk in chunks]  # 병렬처리...?
        xs = [self.dropout(x) for x in xs]
        out = tf.concat([self.fc2(x) for x in xs], axis=1)
        return out
