import pickle

import numpy as np
import tensorflow as tf

def print_shape(name, tensor):
    print("{} : {}".format(name, tensor.shape))


def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask




def biLSTM(inputs, dim, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Output of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd,
                                                                     inputs=inputs, dtype=tf.float32, scope=name)
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

        H = activation(tf.matmul(x, W) + b, name='activation')  # [batch, out_size]
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
        C = 1.0 - T

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), name='y') # y = (H * T) + (x * C)
        return y

def init_highway(x, activation, high_param, name):
    W_, b_, W_T, b_T = high_param
    with tf.variable_scope(name):
        W = tf.Variable(W_, "{}/W".format(name))
        b = tf.Variable(b_, "{}/b".format(name))

        with tf.variable_scope('transform_gate'):
            W_T = tf.Variable(W_T, "{}/W_T".format(name))
            b_T = tf.Variable(b_T, "{}/b_T".format(name))

        H = activation(tf.matmul(x, W) + b, name='activation')  # [batch, out_size]
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
        C = 1.0 - T

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), name='y') # y = (H * T) + (x * C)
        return y

def seq_xw_plus_b(x, W, b):
    input_shape = tf.shape(x)
    dim, out_dim = W.get_shape().as_list()
    x_flat = tf.reshape(x, shape=[-1, dim])
    z = tf.nn.xw_plus_b(x_flat, W, b)
    out_shape = tf.concat([input_shape[:2],tf.shape(W)[1:]], axis=0)
    out = tf.reshape(z, shape=out_shape)
    return out


def intra_attention(x, name):
    x = tf.cast(x, dtype=tf.float32)
    _, max_seq, dim = x.get_shape().as_list()
    with tf.variable_scope(name):
        W, b = weight_bias([dim, dim], [dim])
        E = tf.nn.relu(seq_xw_plus_b(x, W, b)) # [batch, seq, dim]
        E2 = tf.matmul(E, E, transpose_b=True) # [ batch, seq, seq]
        a_raw = tf.matmul(E2, x) # [ batch, seq, dim]
        a_z = tf.reshape(tf.tile(tf.reduce_sum(E2, axis=2), [1,dim]), tf.shape(x)) #[batch, seq]
        a = tf.div(a_raw, a_z) #[batch, seq, dim]
        return a

def inter_attention(p, h, name):
    _, max_seq, dim = p.get_shape().as_list()
    with tf.variable_scope(name):
        W, b = weight_bias([dim, dim], [dim])
        F_p = tf.nn.relu(seq_xw_plus_b(p, W, b))  # [batch, seq_p, dim]
        F_h = tf.nn.relu(seq_xw_plus_b(h, W, b))  # [batch, seq_h, dim]
        E = tf.matmul(F_h, F_p, transpose_b=True) # [batch, seq_h, seq_p]
        beta_raw = tf.transpose(tf.matmul(p, E, transpose_a=True, transpose_b=True),[0,2,1]) # [batch, seq_h, dim]
        beta_z = tf.reshape(tf.tile(tf.reduce_sum(E, axis=2),[1,dim]), tf.shape(h)) # [batch, seq_h, dim]
        beta = tf.div(beta_raw, beta_z)
        alpha_raw = tf.transpose(tf.matmul(h, E, transpose_a=True), [0,2,1] ) # [batch, seq_p, dim ]
        alpha_z = tf.reshape(tf.tile(tf.reduce_sum(E, axis=1),[1,dim]), tf.shape(p)) # [batch, seq_p, dim]
        alpha = tf.div(alpha_raw, alpha_z)

        return alpha, beta

# add s2 's information to s1
def attention(s1, s2, dim1, dim2, dim_inter, name):
    _, max_seq1, _ = s1.get_shape().as_list()
    _, max_seq2, _ = s2.get_shape().as_list()
    with tf.variable_scope(name):
        W1, b1 = weight_bias([dim1, dim_inter], [dim_inter])
        W2, b2 = weight_bias([dim2, dim_inter], [dim_inter])
        F_1 = tf.nn.relu(seq_xw_plus_b(s1, W1, b1))  # [batch, seq_1, dim_inter]
        F_2 = tf.nn.relu(seq_xw_plus_b(s2, W2, b2))  # [batch, seq_2, dim_inter]
        E = tf.matmul(F_1, F_2, transpose_b=True) # [batch, seq_1, seq_2]

        E_soft = tf.nn.softmax(E, dim=2)
        alpha= tf.transpose(tf.matmul(s2, E_soft, transpose_a=True), [0,2,1] ) # [batch, seq_1, dim_2 ]
        return alpha


def interaction_feature(a,b, axis):
    f_concat = tf.concat([a, b], axis=axis)
    f_sub = a - b
    f_odot = tf.multiply(a, b)
    return f_concat, f_sub, f_odot


def dense(input, input_size, output_size, name):
    with tf.variable_scope(name):
        W = tf.get_variable(
            "W",
            shape=[input_size, output_size],
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[output_size]), name="b")
        output = tf.nn.xw_plus_b(input, W, b)  # [batch, num_class]
        return output


def init_dense(input, param, name):
    W_, b_ = param
    with tf.variable_scope(name):
        W = tf.Variable(W_)
        b = tf.Variable(b_)
        output = tf.nn.xw_plus_b(input, W, b)  # [batch, num_class]
        return output


def cartesian(v1, v2):
    # [d1] [d2], -> [d1,d2] [d2,d1]
    _, d1, = v1.get_shape().as_list()
    _, d2, = v2.get_shape().as_list()
    v1_e = tf.tile(v1, [1,d2]) # [batch*seq, d1*d2]
    v1_flat = tf.reshape(v1_e, [-1, d1, d2])
    v2_e = tf.tile(v2, [1,d1]) #
    v2_flat = tf.reshape(v2_e, [-1, d2, d1])
    return tf.matmul(v1_flat, v2_flat) # [batch*seq, d1, d1]


def factorization_machine(input, input_size, name):
    # input : [ -1, input_size]
    hidden_dim = args.fm_latent
    with tf.variable_scope(name):
        L = tf.reshape(dense(input, input_size, 1, "w"), [-1]) # [batch*seq]

        v = tf.get_variable(
            "v",
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            shape=[input_size, hidden_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        P1 = tf.pow(tf.matmul(input, v), 2)
        P2 = tf.matmul(tf.pow(input, 2), tf.pow(v, 2))
        P = tf.multiply(0.5,
            tf.reduce_sum(P1-P2,1))
        return L + P



def LSTM_pool(input, max_seq, input_len, dim, dropout_keep_prob, name):
    with tf.variable_scope(name):
        lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=dim)


    batch_max_len = tf.cast(tf.reduce_max(input_len), dtype=tf.int32)
    input_crop = input[:, :batch_max_len,:]
    #hidden_states, cell_states = tf.nn.dynamic_rnn(cell=lstm_fwd,
    #                                               inputs=input_crop,
    #                                               dtype=tf.float32,
    #                                               scope=name)
    hidden_states, cell_states = lstm_cell(input_crop, dtype=tf.float32, scope=name)
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

    return tf.reshape(tf.concat([max_pooled, avg_pooled], axis=2),[-1, dim*2], name="encode_{}".format(name))

