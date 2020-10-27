import tensorflow as tf


def lm_cross_entropy_loss(true, logits):
    # true : [batch, seq_len]
    # logits : [batch, seq_len, vocab_size]
    vocab_size = logits.shape[-1]
    shift_logits = logits[:, :-1, :]
    shift_logits = tf.reshape(shift_logits, shape=[-1, vocab_size])

    shift_true = true[:, 1:]
    shift_true = tf.reshape(shift_true, shape=[-1])
    padding_mask = tf.cast(tf.equal(shift_true, 0), tf.float32)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        shift_true, shift_logits, from_logits=True
    )
    loss *= 1. - padding_mask
    return tf.reduce_sum(loss) / tf.reduce_sum(1.-padding_mask)
