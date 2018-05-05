import tensorflow as tf
from models.common import dense, factorization_machine, highway_layer, \
    inter_attention, interaction_feature, intra_attention, LSTM_pool, \
    POS_Tagging
from parameter import args


def cafe_network(input_p, input_h, input_p_pos, input_h_pos,
                 input_p_exact, input_h_exact, input_p_char, input_h_char,
                 batch_size, num_classes, embedding, emb_size,
                 max_seq, l2_loss, dropout_keep_prob, is_train):
    """Cafe Network initializer."""
    print("CAFE network..")
    if args.use_char_emb:
        emb_size += args.max_sequence
    if args.syntactical_features:
        emb_size += len(POS_Tagging) + 1
    highway_size = emb_size

    def highway_layer_batch_double(x, seq_len, in_size, out_size, name):
        activation = tf.nn.relu
        x_flat = tf.reshape(x, [-1, in_size])
        h1 = highway_layer(x_flat, in_size, out_size, activation,
                           "{}/high1".format(name))
        h2 = highway_layer(h1, out_size, out_size, activation,
                           "{}/high2".format(name))
        return tf.reshape(h2, [-1, seq_len, out_size])

    def encode(sent, s_len, name):
        # [batch, max_seq, dim]
        embedded = tf.reshape(sent, [-1, emb_size])
        h = highway_layer(embedded, emb_size, tf.nn.relu,
                          "{}/high1".format(name))
        h2 = highway_layer(h, emb_size, tf.nn.relu, "{}/high2".format(name))
        h2_drop = tf.nn.dropout(h2, dropout_keep_prob)
        h_out = tf.reshape(h2_drop, [-1, s_len, emb_size])
        att = intra_attention(h_out, name)

        return h_out, att

    def batch_FM(input_feature, s_len, input_size, l2_loss, name):
        max_len = tf.cast(tf.reduce_max(s_len), dtype=tf.int32)
        input_crop = input_feature[:, :max_len, :]
        input_flat = tf.reshape(input_crop, [-1, input_size])
        n_item = batch_size * max_len
        fm_flat = factorization_machine(input_flat, n_item, input_size,
                                        l2_loss, name)
        return tf.reshape(fm_flat, [-1, max_len])

    def align_fm(s, sp, s_len, name):
        # s, sp : [batch, s_len, ?]
        f_concat, f_sub, f_odot = interaction_feature(s, sp, axis=2)
        v_1 = batch_FM(f_concat, s_len, highway_size*2, l2_loss,
                       "{}/concat".format(name))
        v_2 = batch_FM(f_sub, s_len, highway_size, l2_loss,
                       "{}/sub".format(name))
        v_3 = batch_FM(f_odot, s_len, highway_size, l2_loss,
                       "{}/odot".format(name))
        return tf.stack([v_1, v_2, v_3], axis=2)

    # Function for embedding lOokup and dropout at embedding layer
    def emb_drop(E, x):
        emb = tf.nn.embedding_lookup(E, x)
        return dropout(emb, dropout_keep_prob, is_train)

    def dropout(x, keep_prob, is_training, noise_shape=None,
                seed=None, name=None):
        with tf.name_scope(name or "dropout"):
            # if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_training, lambda: d, lambda: x)
            return out

    def conv1d(in_, filter_size, height, padding, is_train=None,
               keep_prob=1.0, scope=None):
        with tf.variable_scope(scope or "conv1d"):
            num_channels = in_.get_shape()[-1]
            filter_ = tf.get_variable(
                "filter", shape=[1, height, num_channels, filter_size],
                dtype='float')
            bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
            strides = [1, 1, 1, 1]
            # if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
            # [N*M, JX, W/filter_stride, d]
            xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
            out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
            return out

    def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None,
                     keep_prob=1.0, scope=None):
        with tf.variable_scope(scope or "multi_conv1d"):
            assert len(filter_sizes) == len(heights)
            outs = []
            for filter_size, height in zip(filter_sizes, heights):
                if filter_size == 0:
                    continue
                out = conv1d(in_, filter_size, height, padding,
                             is_train=is_train, keep_prob=keep_prob,
                             scope="conv1d_{}".format(height))
                outs.append(out)
            # concat_out = tf.concat(2, outs)
            concat_out = tf.concat(outs, axis=2)
            return concat_out

    with tf.device('/gpu:0'):
        # preprocessing
        # Embedding layer #
        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                premise = emb_drop(embedding, input_p)     # P
                hypothesis = emb_drop(embedding, input_h)  # H

        if args.use_char_emb:
            print("Using Charecter Features")
            # char features
            with tf.variable_scope("char_emb"):
                char_emb_mat = tf.get_variable(
                    "char_emb_mat",
                    shape=[args.char_vocab_size, args.char_emb_size])

                with tf.variable_scope("char") as scope:
                    char_pre = tf.nn.embedding_lookup(char_emb_mat,
                                                      input_p_char)
                    char_hyp = tf.nn.embedding_lookup(char_emb_mat,
                                                      input_h_char)

                    filter_sizes = list(                            # [100]
                        map(int, args.out_channel_dims.split(',')))
                    # [5]
                    heights = list(map(int, args.filter_heights.split(',')))
                    assert sum(filter_sizes) == args.char_out_size, (
                        filter_sizes, args.char_out_size)

                    with tf.variable_scope("conv") as scope:
                        conv_pre = multi_conv1d(char_pre, filter_sizes,
                                                heights, "VALID", is_train,
                                                args.keep_rate, scope='conv')
                        scope.reuse_variables()
                        conv_hyp = multi_conv1d(char_hyp, filter_sizes,
                                                heights, "VALID", is_train,
                                                args.keep_rate, scope='conv')

                        conv_pre = tf.reshape(conv_pre, [-1, max_seq,
                                                         args.char_out_size])
                        conv_hyp = tf.reshape(conv_hyp, [-1, max_seq,
                                                         args.char_out_size])
                premise = tf.concat([premise, conv_pre], axis=2)
                hypothesis = tf.concat([hypothesis, conv_hyp], axis=2)

        if args.syntactical_features:
            print("Using syntactical features")
            # syntactical feature - POS
            premise = tf.concat((premise,
                                 tf.cast(input_p_pos, tf.float32)), axis=2)
            hypothesis = tf.concat((hypothesis,
                                    tf.cast(input_h_pos, tf.float32)), axis=2)

            # syntactical feature - binary exact match
            premise = tf.concat(
                [premise, tf.cast(input_p_exact, tf.float32)], axis=2)
            hypothesis = tf.concat(
                [hypothesis, tf.cast(input_h_exact, tf.float32)], axis=2)

        with tf.device('/cpu:0'):
            data_len = tf.cast(tf.reduce_max(args.max_sequence),
                               dtype=tf.int32)

#  actual network
        # [batch, s_len, dim*2]
        p_enc1, p_intra_att = encode(premise[:, :data_len],
                                     data_len, "premise")

        # [batch, s_len, 3]
        p_intra = align_fm(p_enc1, p_intra_att,
                           args.max_sequence, "premise_intra")

        h_enc1, h_intra_att = encode(hypothesis[:, :data_len],
                                     data_len, "hypothesis")
        h_intra = align_fm(h_enc1, h_intra_att,
                           args.max_sequence, "hypothesis_intra")

        alpha, beta = inter_attention(p_enc1, h_enc1, "inter_attention")
        p_inter = align_fm(p_enc1, alpha, args.max_sequence, "premise_inter")
        h_inter = align_fm(h_enc1, beta, args.max_sequence, "hypothesis_inter")

        p_combine = tf.concat([p_enc1, p_intra, p_inter], axis=2)
        h_combine = tf.concat([h_enc1, h_intra, h_inter], axis=2)

        # h_intra has 3 elem, h_inter has 3 elem
        encode_width = highway_size + 6

        # [batch, dim*2+3]
        p_encode = LSTM_pool(p_combine, max_seq, args.max_sequence,
                             encode_width, dropout_keep_prob, "p_lstm")
        # [batch, dim*2+3]
        h_encode = LSTM_pool(h_combine, max_seq, args.max_sequence,
                             encode_width, dropout_keep_prob, "h_lstm")

        f_concat, f_sub, f_odot = interaction_feature(p_encode, h_encode,
                                                      axis=1)

        feature = tf.concat([f_concat, f_sub, f_odot], axis=1, name="feature")
        h_width = 4*(encode_width*2)
        h = highway_layer(feature, h_width, tf.nn.relu, "pred/high1")
        h2 = highway_layer(h, h_width, tf.nn.relu, "pred/high2")
        h2_drop = tf.nn.dropout(h2, dropout_keep_prob)
        y_pred = dense(h2_drop, h_width, num_classes, l2_loss, "pred/dense")
    return y_pred
