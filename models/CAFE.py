from models.common import *


def cafe_network(input_p, input_h, input_p_len, input_h_len, batch_size, num_classes, embedding, emb_size, max_seq, dropout_keep_prob):
    print("CAFE network..")
    highway_size = emb_size
    with tf.device('/cpu:0'):
        p_len = tf.cast(tf.reduce_max(input_p_len), dtype=tf.int32)
        h_len = tf.cast(tf.reduce_max(input_h_len), dtype=tf.int32)

    def highway_layer_batch_double(x, seq_len, in_size, out_size, name):
        activation = tf.nn.relu
        x_flat = tf.reshape(x, [-1, in_size])
        h1 = highway_layer(x_flat, in_size, out_size, activation, "{}/high1".format(name))
        h2 = highway_layer(h1, out_size, out_size, activation, "{}/high2".format(name))
        return tf.reshape(h2, [-1, seq_len, out_size])

    def encode(sent, s_len, name):
        embedded_raw = tf.nn.embedding_lookup(embedding, sent, name=name)  # [batch, max_seq, dim]
        embedded = tf.reshape(embedded_raw, [-1, emb_size])
        h = highway_layer(embedded, emb_size, tf.nn.relu, "{}/high1".format(name))
        h2 = highway_layer(h, emb_size, tf.nn.relu, "{}/high2".format(name))
        h2_drop = tf.nn.dropout(h2, dropout_keep_prob)
        h_out = tf.reshape(h2_drop, [-1, s_len, emb_size])
        att = intra_attention(h_out, name)

        return h_out, att

    def batch_FM(input_feature, s_len, input_size, name):
        max_len = tf.cast(tf.reduce_max(s_len),dtype=tf.int32)
        input_crop = input_feature[:, :max_len, :]
        input_flat = tf.reshape(input_crop, [-1, input_size])
        fm_flat = factorization_machine(input_flat, input_size, name)
        return tf.reshape(fm_flat, [-1, max_len])

    def align_fm(s, sp, s_len, name):
        # s, sp : [batch, s_len, ?]
        f_concat, f_sub, f_odot = interaction_feature(s, sp, axis=2)
        v_1 = batch_FM(f_concat, s_len, highway_size*2, "{}/concat".format(name))
        v_2 = batch_FM(f_sub, s_len, highway_size, "{}/sub".format(name))
        v_3 = batch_FM(f_odot, s_len, highway_size, "{}/odot".format(name))
        return tf.stack([v_1, v_2, v_3], axis=2)

    with tf.device('/gpu:0'):
        p_enc1, p_intra_att = encode(input_p[:, :p_len], p_len, "premise")  # [batch, s_len, dim*2]
        p_intra = align_fm(p_enc1, p_intra_att, input_p_len, "premise_intra") # [batch, s_len, 3]

    with tf.device('/gpu:0'):
        h_enc1, h_intra_att = encode(input_h[:, :h_len], h_len, "hypothesis")
        h_intra = align_fm(h_enc1, h_intra_att, input_h_len, "hypothesis_intra")

        alpha, beta = inter_attention(p_enc1, h_enc1, "inter_attention")
        p_inter = align_fm(p_enc1, alpha, input_p_len, "premise_inter")
        h_inter = align_fm(h_enc1, beta, input_h_len, "hypothesis_inter")


        p_combine = tf.concat([p_enc1, p_intra, p_inter], axis=2)
        h_combine = tf.concat([h_enc1, h_intra, h_inter], axis=2)

        encode_width = highway_size + 6  # h_intra has 3 elem, h_inter has 3 elem

        p_encode = LSTM_pool(p_combine, max_seq, input_p_len, encode_width, dropout_keep_prob, "p_lstm") # [batch, dim*2+3]
        h_encode = LSTM_pool(h_combine, max_seq , input_h_len, encode_width, dropout_keep_prob, "h_lstm") # [batch, dim*2+3]

        f_concat, f_sub, f_odot = interaction_feature(p_encode, h_encode, axis=1)

        feature = tf.concat([f_concat, f_sub, f_odot], axis=1, name="feature")
        h_width = 4*(encode_width*2)
        h = highway_layer(feature, h_width, tf.nn.relu, "pred/high1")
        h2 = highway_layer(h, h_width, tf.nn.relu, "pred/high2")
        h2_drop = tf.nn.dropout(h2, dropout_keep_prob)
        y_pred = dense(h2_drop, h_width, num_classes, "pred/dense")
    return y_pred


def cafe_direct_emb(p_emb, h_emb, input_p_len, input_h_len, num_classes, embedding, emb_size, max_seq, dropout_keep_prob):
    print("CAFE network..")
    highway_size = emb_size
    with tf.device('/cpu:0'):
        p_len = tf.cast(tf.reduce_max(input_p_len), dtype=tf.int32)
        h_len = tf.cast(tf.reduce_max(input_h_len), dtype=tf.int32)


    def encode(sent, s_len, name):
        embedded = tf.reshape(sent, [-1, emb_size])
        h = highway_layer(embedded, emb_size, tf.nn.relu, "{}/high1".format(name))
        h2 = highway_layer(h, emb_size, tf.nn.relu, "{}/high2".format(name))
        h2_drop = tf.nn.dropout(h2, dropout_keep_prob)
        h_out = tf.reshape(h2_drop, [-1, s_len, emb_size])
        att = intra_attention(h_out, name)

        return h_out, att

    def batch_FM(input_feature, s_len, input_size, name):
        max_len = tf.cast(tf.reduce_max(s_len),dtype=tf.int32)
        input_crop = input_feature[:, :max_len, :]
        input_flat = tf.reshape(input_crop, [-1, input_size])
        fm_flat = factorization_machine(input_flat, input_size, name)
        return tf.reshape(fm_flat, [-1, max_len])

    def align_fm(s, sp, s_len, name):
        # s, sp : [batch, s_len, ?]
        f_concat, f_sub, f_odot = interaction_feature(s, sp, axis=2)
        v_1 = batch_FM(f_concat, s_len, highway_size*2, "{}/concat".format(name))
        v_2 = batch_FM(f_sub, s_len, highway_size, "{}/sub".format(name))
        v_3 = batch_FM(f_odot, s_len, highway_size, "{}/odot".format(name))
        return tf.stack([v_1, v_2, v_3], axis=2)

    with tf.device('/gpu:0'):
        p_enc1, p_intra_att = encode(p_emb, p_len, "premise")  # [batch, s_len, dim*2]
        p_intra = align_fm(p_enc1, p_intra_att, input_p_len, "premise_intra") # [batch, s_len, 3]

    with tf.device('/gpu:0'):
        h_enc1, h_intra_att = encode(h_emb, h_len, "hypothesis")
        h_intra = align_fm(h_enc1, h_intra_att, input_h_len, "hypothesis_intra")

        alpha, beta = inter_attention(p_enc1, h_enc1, "inter_attention")
        p_inter = align_fm(p_enc1, alpha, input_p_len, "premise_inter")
        h_inter = align_fm(h_enc1, beta, input_h_len, "hypothesis_inter")


        p_combine = tf.concat([p_enc1, p_intra, p_inter], axis=2)
        h_combine = tf.concat([h_enc1, h_intra, h_inter], axis=2)

        encode_width = highway_size + 6  # h_intra has 3 elem, h_inter has 3 elem

        p_encode = LSTM_pool(p_combine, max_seq, input_p_len, encode_width, dropout_keep_prob, "p_lstm") # [batch, dim*2+3]
        h_encode = LSTM_pool(h_combine, max_seq , input_h_len, encode_width, dropout_keep_prob, "h_lstm") # [batch, dim*2+3]

        f_concat, f_sub, f_odot = interaction_feature(p_encode, h_encode, axis=1)

        feature = tf.concat([f_concat, f_sub, f_odot], axis=1, name="feature")
        h_width = 4*(encode_width*2)

        if False:
            y_pred = polynom_layer(feature, h_width)
        else:
            h = highway_layer(feature, h_width, tf.nn.relu, "pred/high1")
            h2 = highway_layer(h, h_width, tf.nn.relu, "pred/high2")
            h2_drop = tf.nn.dropout(h2, dropout_keep_prob)
            y_pred = dense(h2_drop, h_width, num_classes, "pred/dense")
    return y_pred




def feature_classification(feature, high1_param, high2_param, dense_param):
    #f_concat, f_sub, f_odot = interaction_feature(p_encode, h_encode, axis=1)
    #feature = tf.concat([f_concat, f_sub, f_odot], axis=1, name="feature")
    h = init_highway(feature, tf.nn.relu, high1_param, "pred/high1")
    h2 = init_highway(h, tf.nn.relu, high2_param, "pred/high1")
    y_pred = init_dense(h2, dense_param, "pred/dense")
    return y_pred
