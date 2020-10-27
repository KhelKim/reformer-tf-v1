import tensorflow as tf
from .embeddings import Embeddings
from .decoder import Decoder
layers = tf.keras.layers


class Reformer(tf.keras.Model):
    def __init__(self, hp, use_rev=True):
        super(Reformer, self).__init__()
        self.embed = Embeddings(hp)
        self.decoder = Decoder(hp)
        self.out_proj = layers.Dense(hp.data.vocab_size, use_bias=False)
        self.n_chunks = hp.model.n_chunks
        self.use_rev = use_rev

    def call(self, x):
        self.padding_mask = tf.cast(tf.equal(x, 0), dtype=tf.float32)
        self.embed_out = self.embed(x)
        if self.use_rev:
            self.decoder_out = tf.stop_gradient(
                self.decoder(self.embed_out, self.embed_out, padding_mask=self.padding_mask))
        else:
            self.decoder_out = self.decoder(self.embed_out, self.embed_out, padding_mask=self.padding_mask)
        chunks = tf.split(self.decoder_out, num_or_size_splits=self.n_chunks, axis=1)
        out = tf.concat([self.out_proj(chunk) for chunk in chunks], axis=1)
        return out

    def compute_gradients(self, out, ys):
        grads_and_vars = []

        inp, variables, out = self.decoder_out, self.out_proj.trainable_weights, out
        inp_n_variables = [inp] + variables
        gradients = tf.gradients(out, inp_n_variables, grad_ys=ys)
        gradients = [tf.stop_gradient(gradient) for gradient in gradients]
        ys, gradients = gradients[0], gradients[1:]
        grads_and_vars.extend(zip(gradients, variables))

        if self.use_rev is True:
            ys, grads_and_vars = self.decoder.backward_grads(ys, self.padding_mask)
            grads_and_vars.extend(grads_and_vars)
        else:
            inp, variables, out = self.embed_out, self.decoder.trainable_weights, self.decoder_out
            inp_n_variables = [inp] + variables
            gradients = tf.gradients(out, inp_n_variables, grad_ys=ys)
            gradients = [tf.stop_gradient(gradient) for gradient in gradients]
            ys, gradients = gradients[0], gradients[1:]
            grads_and_vars.extend(zip(gradients, variables))

        variables, out = self.embed.trainable_weights, self.embed_out
        gradients = tf.gradients(out, variables, grad_ys=ys)
        gradients = [tf.stop_gradient(gradient) for gradient in gradients]
        grads_and_vars.extend(zip(gradients, variables))

        return grads_and_vars
