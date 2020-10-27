import tensorflow as tf
from .attentions import MultiRoundsLSAttention
from .chunkfeedforward import ChunkFeedForward
layers = tf.keras.layers


class Block(layers.Layer):
    def __init__(self, hp, func):
        super(Block, self).__init__()
        self.func = func
        self.dropout = layers.Dropout(hp.model.dropout_rate)
        self.norm = layers.LayerNormalization(epsilon=1e-12)

    def call(self, x, random=True, padding_mask=None):
        x = self.func(x, random=random, padding_mask=padding_mask)
        x = self.dropout(x)
        out = self.norm(x)
        return out


class ReversibleDecoderLayer(layers.Layer):
    def __init__(self, hp):
        super(ReversibleDecoderLayer, self).__init__()
        self.attn = MultiRoundsLSAttention(hp)
        self.feed_forward = ChunkFeedForward(hp)
        self.f_block = Block(hp, self.attn)
        self.g_block = Block(hp, self.feed_forward)

    def call(self, x1, x2, padding_mask=None):
        y1 = x1 + self.f_block(x2, padding_mask=padding_mask)
        y2 = x2 + self.g_block(y1)
        return y1, y2

    def backward_grads(self, ys, dys, padding_mask=None):
        y1, y2 = ys
        y1, y2 = tf.stop_gradient(y1), tf.stop_gradient(y2)
        dy1, dy2 = dys

        gy1 = self.g_block(y1)
        grads_combines = tf.gradients(
            gy1, [y1] + self.g_block.trainable_weights, grad_ys=dy2)

        dg = grads_combines[1:]
        dx1 = dy1 + grads_combines[0]
        x2 = y2 - gy1

        fx2 = self.f_block(x2, random=False, padding_mask=padding_mask)
        grads_combines = tf.gradients(
            fx2, [x2] + self.f_block.trainable_variables, grad_ys=dx1)
        df = grads_combines[1:]
        dx2 = dy2 + grads_combines[0]
        x1 = y1 - fx2
        grads_and_vars = [(g, v) for g, v in zip(df + dg,
                                                 self.f_block.trainable_variables + self.g_block.trainable_weights)]

        return (x1, x2), (dx1, dx2), grads_and_vars


class Decoder(layers.Layer):
    def __init__(self, hp):
        super(Decoder, self).__init__()
        self.layers = [ReversibleDecoderLayer(hp) for _ in range(hp.model.n_layers)]

    def call(self, x1, x2, padding_mask=None):
        for layer in self.layers:
            x1, x2 = layer(x1, x2, padding_mask)
        self.results = x1, x2
        return x1 + x2

    def backward_grads(self, dy, padding_mask=None):
        total_grads_and_vars = []

        dys = dy, dy
        ys = self.results

        for layer in self.layers[::-1]:
            ys, dys, grads_and_vars = layer.backward_grads(ys, dys, padding_mask=padding_mask)
            total_grads_and_vars.extend(grads_and_vars)

        return dys[0] + dys[1], total_grads_and_vars
