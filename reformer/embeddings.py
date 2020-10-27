import tensorflow as tf
layers = tf.keras.layers


class PositionEmbeddings(layers.Layer):
    def __init__(self, hp):
        super(PositionEmbeddings, self).__init__()
        self.embedding = layers.Embedding(hp.model.max_position_embeddings, hp.model.d_model)
        self.dropout = layers.Dropout(hp.model.dropout_rate)

    def call(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = self.dropout(position_embeddings)
        return position_embeddings


class AxialPositionEmbeddings(layers.Layer):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_reformer.py#L115-L206

    def __init__(self, hp):
        raise NotImplemented


class Embeddings(layers.Layer):
    def __init__(self, hp):
        super(Embeddings, self).__init__()
        self.max_position_embeddings = hp.model.max_position_embeddings
        self.word_embeddings = layers.Embedding(hp.data.vocab_size, hp.model.d_model)
        self.dropout1 = layers.Dropout(hp.model.dropout_rate)
        self.dropout2 = layers.Dropout(hp.model.dropout_rate)
        self.position_embeddings = (
            AxialPositionEmbeddings(hp) if hp.model.axial_pos_embds else PositionEmbeddings(hp)
        )

    def call(self, x):
        input_shape = tf.shape(x)
        seq_len = input_shape[1]

        # [batch, seq_len]
        position_ids = tf.broadcast_to(tf.range(seq_len), input_shape)

        x = self.word_embeddings(x)
        x = self.dropout1(x)
        x += self.position_embeddings(position_ids)
        x = self.dropout2(x)
        return x
