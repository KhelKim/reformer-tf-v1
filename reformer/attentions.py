import math
import tensorflow as tf
layers = tf.keras.layers


# helpers
def expand_w_one_chunk_bucket(x):
    # x : [batch * heads, n_buckets, bucket_length, d_k, rounds]
    shift = tf.concat([x[:, -1:], x[:, :-1]], axis=1)
    concat = tf.concat([shift, x], axis=2)
    return concat


def same_size_gather_axis1(origin, indices):
    # original : [batch*heads, seq_len, n_rounds, ...]
    # indices : [batch*heads, seq_len, n_rounds]
    input_shape = tf.shape(indices)
    b_m_h = input_shape[0]
    seq_len = input_shape[1]
    n_rounds = input_shape[2]

    indices = tf.reshape(indices, [b_m_h, n_rounds * seq_len])[:, :, tf.newaxis]

    bat_m_heads_idx = tf.range(b_m_h)[:, tf.newaxis]
    bat_m_heads_idx = tf.broadcast_to(bat_m_heads_idx, shape=[b_m_h, n_rounds * seq_len])[:, :, tf.newaxis]

    n_rounds_idx = tf.range(n_rounds)[tf.newaxis, tf.newaxis, :]
    n_rounds_idx = tf.broadcast_to(n_rounds_idx, shape=[b_m_h, seq_len, n_rounds])
    n_rounds_idx = tf.reshape(n_rounds_idx, shape=[b_m_h, n_rounds * seq_len])[:, :, tf.newaxis]

    indices = tf.concat([bat_m_heads_idx, indices, n_rounds_idx], axis=-1)
    indices = tf.reshape(indices, shape=[b_m_h, seq_len, n_rounds, 3])
    out = tf.gather_nd(origin, indices)
    return out


def same_size_gather_axis2(origin, indices):
    # original : [batch*heads, seq_len, 2*bucket_len*n_rounds]
    # indices : [batch*heads, seq_len, 2*bucket_len*n_rounds]
    input_shape = tf.shape(indices)
    b_m_h = input_shape[0]
    seq_len = input_shape[1]
    bl_m_r_m_2 = input_shape[2]

    indices = tf.reshape(indices, (b_m_h*seq_len, bl_m_r_m_2, 1))

    batch_idx = tf.broadcast_to(tf.range(b_m_h)[:, tf.newaxis], shape=[b_m_h, seq_len])
    batch_idx = tf.reshape(batch_idx, (b_m_h*seq_len, 1, 1))
    batch_idx = tf.broadcast_to(batch_idx, (b_m_h*seq_len, bl_m_r_m_2, 1))

    seq_idx = tf.broadcast_to(tf.range(seq_len)[tf.newaxis, :], shape=[b_m_h, seq_len])
    seq_idx = tf.reshape(seq_idx, (b_m_h*seq_len, 1, 1))
    seq_idx = tf.broadcast_to(seq_idx, (b_m_h*seq_len, bl_m_r_m_2, 1))

    indices = tf.concat([batch_idx, seq_idx, indices], axis=-1)
    indices = tf.reshape(indices, shape=[b_m_h, seq_len, bl_m_r_m_2, 3])
    out = tf.gather_nd(origin, indices)
    return out


def query_value_gather(origin, indices):
    # original : [batch*heads, seq_len, d_k]
    # indices : [batch*heads, seq_len, n_rounds]
    input_shape = tf.shape(origin)
    b_m_h = input_shape[0]
    seq_len = input_shape[1]
    d_k = input_shape[2]
    n_rounds = indices.shape[2]

    # [batch*heads*seq_len, d_k]
    origin = tf.reshape(origin, [b_m_h * seq_len, d_k])
    # [batch*heads*seq_len, n_rounds]
    indices = tf.reshape(indices, [b_m_h * seq_len, n_rounds])

    # [batch*heads*seq_len, ]
    batch_idx = tf.range(b_m_h, dtype=tf.int32) * seq_len
    batch_idx = tf.broadcast_to(batch_idx[:, tf.newaxis], shape=[b_m_h, seq_len])
    batch_idx = tf.reshape(batch_idx, [-1])[:, tf.newaxis]

    indices += batch_idx

    # [batch*heads*seq_len, n_rounds, d_k]
    gathered_tensor = tf.nn.embedding_lookup(origin, indices)
    # [batch*heads*seq_len, d_k, n_rounds]
    gathered_tensor = tf.transpose(gathered_tensor, perm=[0, 2, 1])
    # [batch*heads, seq_len, d_k, n_rounds]
    gathered_tensor = tf.reshape(gathered_tensor, (b_m_h, seq_len, d_k, n_rounds))
    return gathered_tensor


def get_reverse_indices(indices, axis):
    reverse_indices = tf.argsort(indices, axis=axis)
    return reverse_indices


def count_dup_keys(inp, rounds):
    # inp : [batch*heads, seq, 2*bucket_len*rounds]
    input_shape = tf.shape(inp)
    b_m_h = input_shape[0]
    seq_len = input_shape[1]
    b_l_r_2 = input_shape[2]

    sorting_indices = tf.argsort(inp, axis=2)
    sorted_flat_key = same_size_gather_axis2(inp, sorting_indices)
    unsorting_indices = get_reverse_indices(sorting_indices, axis=2)

    count_shift_keys = tf.ones(shape=[b_m_h, seq_len, b_l_r_2])
    for i in range(1, rounds):
        equiv_flat_key = tf.cast(tf.equal(sorted_flat_key[..., i:], sorted_flat_key[..., :-i]), dtype=tf.float32)
        pad = tf.zeros(shape=[b_m_h, seq_len, i], dtype=tf.float32)
        count_shift_keys += tf.concat([pad, equiv_flat_key], axis=-1)
        count_shift_keys += tf.concat([equiv_flat_key, pad], axis=-1)
    return same_size_gather_axis2(count_shift_keys, unsorting_indices)


class LSHAttention(layers.Layer):
    def __init__(self, hp):
        super(LSHAttention, self).__init__()
        self.d_k = hp.model.d_k
        self.n_heads = hp.model.n_heads
        self.n_rounds = hp.model.n_rounds
        self.dropout = layers.Dropout(hp.model.dropout_rate)
        self.bucket_len = hp.model.bucket_len

        self.batch_size = hp.training.batch_size

        self.rand_matrix = None

    def make_hashes(self, x, n_buckets, b_m_h, random=True):
        # x : [batch*heads, seq_len, d_k]
        seq_len = x.shape[1]

        if random:
            random_matrix_size = [b_m_h, self.d_k, self.n_rounds, n_buckets // 2]
            self.rand_matrix = tf.random.normal(shape=random_matrix_size)

        # [batch*head, seq_len, n_rounds, n_buckets // 2]
        projected_x = tf.einsum("...ij,...jkl->...ikl", x, self.rand_matrix)

        # [batch*head, seq_len, n_rounds]
        hashes = tf.argmax(tf.concat([projected_x, -projected_x], axis=-1), axis=-1, output_type=tf.int32)

        seq_idx = tf.range(seq_len, dtype=hashes.dtype)

        hashes = hashes * seq_len + seq_idx[tf.newaxis, :, tf.newaxis]
        # hashes // seq_len == bucket idx
        # hashes % seq_len == seq_idx
        return hashes

    def call(self, query, value, random=True, padding_mask=0):
        # b_m_h = query.shape[0]  # batch * heads
        input_shape = tf.shape(query)
        b_m_h = input_shape[0]
        seq_len = query.shape[1]
        assert seq_len % (self.bucket_len * 2) == 0
        n_buckets = seq_len // self.bucket_len

        # making hashes and getting sorting indices, unsorting indices
        # [batch*heads, seq_len, n_rounds]
        hashes = self.make_hashes(query, n_buckets, b_m_h, random)
        sorting_indices = tf.argsort(hashes, axis=1)
        unsorting_indices = get_reverse_indices(sorting_indices, axis=1)

        del hashes

        # reordering query
        # [batch*heads, seq_len, d_k, n_rounds]
        query = query_value_gather(query, sorting_indices)

        query = tf.reshape(query,
                           shape=(-1, n_buckets, self.bucket_len,
                                  self.d_k, self.n_rounds))

        # get key
        # [batch*heads, n_buckets, bucket_len*2, d_k, n_rounds]
        key_w_one_chunk_back = expand_w_one_chunk_bucket(query)
        key_w_one_chunk_back = tf.math.l2_normalize(key_w_one_chunk_back, axis=-2)

        # compute qk
        # [batch*heads, n_buckets, bucket_len, bucket_len*2, n_rounds]
        qk = tf.einsum("...ijk,...ljk->...ilk", query, key_w_one_chunk_back)
        qk /= math.sqrt(self.d_k)

        # mask1 : don't look future, mask2 : self attention score is -1e5
        # [batch * head, n_buckets, bucket_len, n_rounds]
        # query_indices = tf.reshape(sorting_indices, shape=(-1, n_buckets, self.bucket_len, self.n_rounds))
        query_indices = tf.reshape(sorting_indices, shape=(-1, n_buckets, self.bucket_len, self.n_rounds)) % seq_len

        # [batch * head, n_buckets, bucket_len*2, rounds]
        key_w_one_chunk_back_indices = expand_w_one_chunk_bucket(query_indices)

        # [batch * head, n_buckets, bucket_len, bucket_len*2, n_rounds]
        mask1 = tf.math.less(query_indices[..., tf.newaxis, :], key_w_one_chunk_back_indices[..., tf.newaxis, :, :])
        mask2 = tf.math.equal(query_indices[..., tf.newaxis, :], key_w_one_chunk_back_indices[..., tf.newaxis, :, :])
        mask1 = tf.cast(mask1, dtype=tf.float32)
        mask2 = tf.cast(mask2, dtype=tf.float32)

        qk = qk * (1-mask1) + (mask1 * -1e9)
        qk = qk * (1-mask2) + (mask2 * -1e5)

        del mask1, mask2

        if tf.is_tensor(padding_mask):
            # padding_mask : [batch_size, max_len], dtype=tf.float32
            batch_size = tf.shape(padding_mask)[0]
            padding_mask = tf.broadcast_to(padding_mask[:, tf.newaxis, :], shape=[batch_size, self.n_heads, seq_len])
            padding_mask = tf.reshape(padding_mask, shape=[-1, seq_len])
            padding_mask = tf.broadcast_to(padding_mask[..., tf.newaxis],
                                           shape=[batch_size*self.n_heads, seq_len, self.n_rounds])
            padding_mask = same_size_gather_axis1(padding_mask, sorting_indices)
            padding_mask = tf.reshape(padding_mask, shape=[-1, n_buckets, self.bucket_len, self.n_rounds])
            padding_mask = expand_w_one_chunk_bucket(padding_mask)
            padding_mask = tf.broadcast_to(padding_mask[:, :, tf.newaxis, :, :],
                                           shape=[batch_size*self.n_heads, n_buckets, self.bucket_len,
                                                  2*self.bucket_len, self.n_rounds])
            qk = qk * (1-padding_mask) + (padding_mask * -1e9)

            del padding_mask

        # count_dup_keys
        # [batch*head, n_buckets, bucket_len, bucket_len*2, n_rounds]
        key_w_one_chunk_back_indices = tf.broadcast_to(key_w_one_chunk_back_indices[:, :, tf.newaxis, :, :],
                                                       shape=[b_m_h, n_buckets, self.bucket_len,
                                                              2 * self.bucket_len, self.n_rounds])
        # [batch*head, seq_len, bucket_len*2*n_rounds]
        # 각 토큰별 같은 버켓에 포함된 토큰들의 인덱스
        key_w_one_chunk_back_indices = tf.reshape(key_w_one_chunk_back_indices, shape=[b_m_h, seq_len,
                                                                                       2*self.bucket_len*self.n_rounds])

        count_keys = count_dup_keys(key_w_one_chunk_back_indices, self.n_rounds)
        count_keys = tf.reshape(count_keys, shape=[b_m_h, seq_len, 2*self.bucket_len, self.n_rounds])

        # get attention score
        qk = tf.reshape(qk, shape=[b_m_h, seq_len, 2*self.bucket_len, self.n_rounds])

        # [batch*head, seq_len, rounds]
        logsumexp_qk = tf.math.reduce_logsumexp(qk, axis=2)

        # [batch*head, seq_len, 2 * bucket_len, rounds]
        softmax_qk = tf.math.exp(qk - logsumexp_qk[..., None, :]) / tf.cast(count_keys, dtype=tf.float32)
        softmax_qk = self.dropout(softmax_qk)

        # processing value
        # [bath*heads, seq_len, d_k, rounds]
        value = query_value_gather(value, sorting_indices)

        # [bath*heads, n_buckets, self.bucket_len, d_k, rounds]
        value = tf.reshape(value, shape=[b_m_h, n_buckets, self.bucket_len, self.d_k, self.n_rounds])

        # [bath*heads, n_buckets, 2*self.bucket_len, d_k, rounds]
        value_w_one_chunk_back = expand_w_one_chunk_bucket(value)

        # get attention
        softmax_qk = tf.reshape(softmax_qk, shape=[b_m_h, n_buckets, self.bucket_len, 2*self.bucket_len, self.n_rounds])

        # [batch*heads, n_buckets, bucket_len, d_k, rounds]
        attention = tf.einsum("...ijl,...jkl->...ikl", softmax_qk, value_w_one_chunk_back)
        attention = tf.reshape(attention, shape=[b_m_h, seq_len, self.d_k, self.n_rounds])
        attention = tf.transpose(attention, perm=[0, 1, 3, 2])
        attention = same_size_gather_axis1(attention, unsorting_indices)
        attention = tf.transpose(attention, perm=[0, 1, 3, 2])

        # exp(z(i, p^r_i) - z(i, p_i)) 12 page, https://arxiv.org/abs/2001.04451
        # [batch*head, seq_len, rounds]
        logsumexp_qk = same_size_gather_axis1(logsumexp_qk, unsorting_indices)
        logsumexp_qk = tf.math.softmax(logsumexp_qk, axis=-1)

        # [batch*head, seq_len, d_k]
        attention = tf.einsum("...ij,...j->...i", attention, logsumexp_qk)
        return attention


class MultiRoundsLSAttention(layers.Layer):
    def __init__(self, hp):
        super(MultiRoundsLSAttention, self).__init__()
        self.d_k = hp.model.d_k
        self.n_heads = hp.model.n_heads
        self.WQ = layers.Dense(hp.model.d_model)
        self.WV = layers.Dense(hp.model.d_model)
        self.WO = layers.Dense(hp.model.d_model)
        self.lsh_attn = LSHAttention(hp)

    def call(self, x, random=True, padding_mask=0):
        # x : [batch, seq_len, d_model]
        seq_len = x.shape[1]

        query = tf.transpose(tf.reshape(self.WQ(x), shape=[-1, seq_len, self.n_heads, self.d_k]), perm=(0, 2, 1, 3))
        value = tf.transpose(tf.reshape(self.WV(x), shape=[-1, seq_len, self.n_heads, self.d_k]), perm=(0, 2, 1, 3))

        query = tf.reshape(query, [-1, seq_len, self.d_k])
        value = tf.reshape(value, [-1, seq_len, self.d_k])

        attention = self.lsh_attn(query, value, random=random, padding_mask=padding_mask)

        attention = tf.transpose(tf.reshape(attention, [-1, self.n_heads, seq_len, self.d_k]), perm=(0, 2, 1, 3))
        attention = tf.reshape(attention, shape=[-1, seq_len, self.n_heads*self.d_k])

        out = self.WO(attention)
        return out

