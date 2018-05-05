import os
import pickle

import numpy as np
import tensorflow as tf

from parameter import args, path_dict

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO',
               ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP',
               "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP',
               'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN',
               'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos: i for i, pos in enumerate(POS_Tagging)}


def avg(l):
    return sum(l)/len(l)


def load_pickle(name):
    path = os.path.join("pickle", name)
    return pickle.load(open(path, "rb"))


def save_pickle(name, obj):
    path = os.path.join("pickle", name)
    return pickle.dump(obj, open(path, "wb"))


def reverse_index(word2idx):
    idx2word = dict()
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return idx2word


def print_shape(name, tensor):
    print("{} : {}".format(name, tensor.shape))


def convert_tokens(tokens, voca):
    OOV = 0
    converted = []
    for t in tokens:
        if t in voca:
            converted.append(voca[t])
        else:
            converted.append(OOV)
        if len(converted) == args.max_sequence:
            break
    while len(converted) < args.max_sequence:
        converted.append(1)
    return np.array(converted), len(tokens)


def length(sequence):
    """
    Get true len of sequences(no padding), and mask for true-len in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask


def load_wemb(word_indice, embedding_size):
    if not os.path.exists("pickle/wemb"):
        return load_embedding(word_indice, embedding_size)

    return tf.Variable(load_pickle("wemb"), trainable=False)


def glove_voca():
    path = path_dict["embedding_data_path"]
    voca = set()
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            voca.add(s[0])
    return voca


def parse_to_pos_vector(parse, left_padding_and_cropping_pair=(0, 0)):
                            # ONE HOT
    pos = parsing_parse(parse)
    pos_vector = [POS_dict.get(tag, 0) for tag in pos]
    left_padding, left_cropping = left_padding_and_cropping_pair
    vector = np.zeros((args.max_sequence, len(POS_Tagging)))
    assert left_padding == 0 or left_cropping == 0

    for i in range(args.max_sequence):
        if i < len(pos_vector):
            vector[i + left_padding, pos_vector[i + left_cropping]] = 1
        else:
            break
    return vector


def generate_pos_feature_tensor(parses, left_padding_and_cropping_pairs):
    pos_vectors = []
    for parse in parses:
        pos = parsing_parse(parse)
        pos_vector = [
            (idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
        pos_vectors.append(pos_vector)

    return construct_one_hot_feature_tensor(pos_vectors,
                                            left_padding_and_cropping_pairs, 2,
                                            column_size=len(POS_Tagging))


def construct_one_hot_feature_tensor(sequences,
                                     left_padding_and_cropping_pairs, dim,
                                     column_size=None, dtype=np.int32):
    """
    sequences: [[(idx, val)... ()]...[]]
    left_padding_and_cropping_pairs: [[(0,0)...] ... []]
    """
    tensor_list = []
    for sequence, pad_crop_pair in zip(sequences,
                                       left_padding_and_cropping_pairs):
        left_padding, left_cropping = pad_crop_pair
        if dim == 1:
            vec = np.zeros((args.max_sequence))
            for num in sequence:
                if num + left_padding - left_cropping < args.max_sequence and \
                 num + left_padding - left_cropping >= 0:
                    vec[num + left_padding - left_cropping] = 1
            tensor_list.append(vec)
        elif dim == 2:
            assert column_size
            mtrx = np.zeros((args.max_sequence, column_size))
            for row, col in sequence:
                if row + left_padding - left_cropping < args.max_sequence and \
                 row + left_padding - left_cropping >= 0 and col < column_size:
                    mtrx[row + left_padding - left_cropping, col] = 1
            tensor_list.append(mtrx)

        else:
            raise NotImplementedError

    return np.array(tensor_list, dtype=dtype)


def fill_feature_vector_with_cropping_or_padding(
     sequences, left_padding_and_cropping_pairs, dim,
     column_size=None, dtype=np.int32):
    """fill a vecoor with padding or crop."""
    if dim == 1:
        list_of_vectors = []
        for sequence, pad_crop_pair in zip(sequences,
                                           left_padding_and_cropping_pairs):
            vec = np.zeros((args.max_sequence))
            left_padding, left_cropping = pad_crop_pair
            for i in range(args.max_sequence):
                if i + left_padding < args.max_sequence and \
                 i - left_cropping < len(sequence):
                    vec[i + left_padding] = sequence[i + left_cropping]
                else:
                    break
            list_of_vectors.append(vec)
        return np.array(list_of_vectors, dtype=dtype)
    elif dim == 2:
        assert column_size
        tensor_list = []
        for sequence, pad_crop_pair in zip(sequences,
                                           left_padding_and_cropping_pairs):
            left_padding, left_cropping = pad_crop_pair
            mtrx = np.zeros((args.max_sequence, column_size))
            for row_idx in range(args.max_sequence):
                if row_idx + left_padding < args.max_sequence and \
                 row_idx < len(sequence) + left_cropping:
                    for (col_idx, content) in enumerate(
                             sequence[row_idx + left_cropping]):
                        mtrx[row_idx + left_padding, col_idx] = content
                else:
                    break
            tensor_list.append(mtrx)
        return np.array(tensor_list, dtype=dtype)
    else:
        raise NotImplementedError


def parsing_parse(parse):
    base_parse = [
        s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
    pos = [pair.split(" ")[0] for pair in base_parse]
    return pos


def get_batches(dataset, start_index, end_index, crop_max=100):
    # data is fully numpy array here
    indices = range(start_index, end_index)

    premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0, 0)] * len(indices)
    p = fill_feature_vector_with_cropping_or_padding(
        [dataset[i]['p'][:] for i in indices], premise_pad_crop_pair, 1)
    h = fill_feature_vector_with_cropping_or_padding(
        [dataset[i]['h'][:] for i in indices], hypothesis_pad_crop_pair, 1)

    p_len = [dataset[i]['p_len'] for i in indices]
    h_len = [dataset[i]['h_len'] for i in indices]

    p_c = [dataset[i]['p_char'][:] for i in indices]
    h_c = [dataset[i]['h_char'][:] for i in indices]

    p_char = fill_feature_vector_with_cropping_or_padding(
        p_c, premise_pad_crop_pair, 2, column_size=args.char_in_word_size)

    h_char = fill_feature_vector_with_cropping_or_padding(
        h_c, hypothesis_pad_crop_pair, 2, column_size=args.char_in_word_size)

    p_pos = generate_pos_feature_tensor(
        [dataset[i]['p_pos'][:] for i in indices], premise_pad_crop_pair)
    h_pos = generate_pos_feature_tensor(
        [dataset[i]['h_pos'][:] for i in indices], hypothesis_pad_crop_pair)

    p_exact = [dataset[i]['p_exact'][:] for i in indices]
    h_exact = [dataset[i]['h_exact'][:] for i in indices]
    y = [dataset[i]['y'] for i in indices]

    return p, h, p_len, h_len, p_pos, h_pos, \
        p_char, h_char, p_exact, h_exact, y


def load_embedding(word_indices, word_embedding_dimension, divident=1.0):
    print("loading embedding")
    path = path_dict["embedding_data_path"]
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = word_embedding_dimension
    emb = np.empty((n, m), dtype=np.float32)

    emb[:, :] = np.random.normal(size=(n, m)) / divident

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1, m), dtype="float32")
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):

            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                    count += 1
                except ValueError:
                    print(s[0])
                    continue

    print("{} of {} initialized".format(count, len(word_indices)))
    save_pickle("wemb", emb)
    return emb


def biLSTM(inputs, dim, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as
    a tuple, and cell states as a tuple.

    Output of hidden states: [(batch_size, max_seq_length, hidden_dim),
        (batch_size, max_seq_length, hidden_dim)]   Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs,
            dtype=tf.float32, scope=name)
    return hidden_states, cell_states


def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.get_variable(
        "weight",
        shape=W_shape,
        regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b


def highway_layer(x, size, activation, name, carry_bias=-1.0):
    with tf.variable_scope(name):
        W, b = weight_bias([size, size], [size])

        with tf.variable_scope('transform_gate'):
            W_T, b_T = weight_bias([size, size], [size], bias_init=carry_bias)

        # [batch, out_size]
        H = activation(tf.matmul(x, W) + b, name='activation')
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
        C = 1.0 - T

        # y = (H * T) + (x * C)
        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), name='y')
        return y


def seq_xw_plus_b(x, W, b):
    input_shape = tf.shape(x)
    dim, out_dim = W.get_shape().as_list()
    x_flat = tf.reshape(x, shape=[-1, dim])
    z = tf.nn.xw_plus_b(x_flat, W, b)
    out_shape = tf.concat([input_shape[:2], tf.shape(W)[1:]], axis=0)
    out = tf.reshape(z, shape=out_shape)
    return out


def intra_attention(x, name):
    x = tf.cast(x, dtype=tf.float32)
    _, max_seq, dim = x.get_shape().as_list()
    with tf.variable_scope(name):
        W, b = weight_bias([dim, dim], [dim])
        E = tf.nn.relu(seq_xw_plus_b(x, W, b))  # [batch, seq, dim]
        E2 = tf.matmul(E, E, transpose_b=True)  # [ batch, seq, seq]
        a_raw = tf.matmul(E2, x)  # [ batch, seq, dim]

        # [batch, seq]
        a_z = tf.reshape(
            tf.tile(tf.reduce_sum(E2, axis=2), [1, dim]), tf.shape(x))
        a = tf.div(a_raw, a_z)  # [batch, seq, dim]
        return a


def inter_attention(p, h, name):
    _, max_seq, dim = p.get_shape().as_list()
    with tf.variable_scope(name):
        W, b = weight_bias([dim, dim], [dim])
        F_p = tf.nn.relu(seq_xw_plus_b(p, W, b))  # [batch, seq_p, dim]
        F_h = tf.nn.relu(seq_xw_plus_b(h, W, b))  # [batch, seq_h, dim]
        E = tf.matmul(F_h, F_p, transpose_b=True)  # [batch, seq_h, seq_p]

        # [batch, seq_h, dim]
        beta_raw = tf.transpose(
            tf.matmul(p, E, transpose_a=True, transpose_b=True), [0, 2, 1])

        # [batch, seq_h, dim]
        beta_z = tf.reshape(
            tf.tile(tf.reduce_sum(E, axis=2), [1, dim]), tf.shape(h))
        beta = tf.div(beta_raw, beta_z)

        # [batch, seq_p, dim]
        alpha_raw = tf.transpose(
            tf.matmul(h, E, transpose_a=True), [0, 2, 1])

        # [batch, seq_p, dim]
        alpha_z = tf.reshape(
            tf.tile(tf.reduce_sum(E, axis=1), [1, dim]), tf.shape(p))
        alpha = tf.div(alpha_raw, alpha_z)

        return alpha, beta


def interaction_feature(a, b, axis):
    f_concat = tf.concat([a, b], axis=axis)
    f_sub = a - b
    f_odot = tf.multiply(a, b)
    return f_concat, f_sub, f_odot


def dense(input, input_size, output_size, l2_loss, name):
    with tf.variable_scope(name):
        W = tf.get_variable(
            "W",
            shape=[input_size, output_size],
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[output_size]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        output = tf.nn.xw_plus_b(input, W, b)  # [batch, num_class]
        return output


def cartesian(v1, v2):
    # [d1] [d2], -> [d1,d2] [d2,d1]
    _, d1, = v1.get_shape().as_list()
    _, d2, = v2.get_shape().as_list()
    v1_e = tf.tile(v1, [1, d2])  # [batch*seq, d1*d2]
    v1_flat = tf.reshape(v1_e, [-1, d1, d2])
    v2_e = tf.tile(v2, [1, d1])
    v2_flat = tf.reshape(v2_e, [-1, d2, d1])
    return tf.matmul(v1_flat, v2_flat)   # [batch*seq, d1, d1]


def factorization_machine(input, n_item, input_size, l2_loss, name):
    # input : [ -1, input_size]
    hidden_dim = 99
    with tf.variable_scope(name):
        # [batch*seq]
        L = tf.reshape(dense(input, input_size, 1, l2_loss, "w"), [-1])

        v = tf.get_variable(
            "v",
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            shape=[input_size, hidden_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        P1 = tf.pow(tf.matmul(input, v), 2)
        P2 = tf.matmul(tf.pow(input, 2), tf.pow(v, 2))
        P = tf.multiply(0.5, tf.reduce_sum(P1-P2, 1))
        return L + P


def LSTM_pool(input, max_seq, input_len, dim, dropout_keep_prob, name):
    with tf.variable_scope(name):
        lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=dim)

    batch_max_len = tf.cast(tf.reduce_max(input_len), dtype=tf.int32)
    input_crop = input[:, :batch_max_len, :]
    # hidden_states, cell_states = tf.nn.dynamic_rnn(cell=lstm_fwd,
    #                                               inputs=input_crop,
    #                                               dtype=tf.float32,
    #                                               scope=name)
    hidden_states, cell_states = lstm_cell(input_crop,
                                           dtype=tf.float32, scope=name)
    # [batch, batch_max_len, dim]
    h_shape = tf.shape(hidden_states)
    hidden_states_drop = tf.nn.dropout(hidden_states, dropout_keep_prob)
    mid = tf.constant(max_seq, shape=[1]) - h_shape[1]
    pad_size = tf.concat([h_shape[0:1], mid, h_shape[2:]], axis=0)
    hidden_states = tf.concat([hidden_states_drop, tf.zeros(pad_size)], axis=1)

    max_pooled = tf.nn.max_pool(
        tf.reshape(hidden_states, [-1, max_seq, dim, 1]),
        ksize=[1, max_seq, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )

    avg_pooled = tf.nn.avg_pool(
        tf.reshape(hidden_states, [-1, max_seq, dim, 1]),
        ksize=[1, max_seq, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )

    return tf.reshape(
        tf.concat([max_pooled, avg_pooled], axis=2),
        [-1, dim*2], name="encode_{}".format(name))
