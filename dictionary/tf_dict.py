import tensorflow as tf

from dictionary.tf_util import *

class DictNN:

    def __init__(self, max_len_var_enc, dim_var, max_len_entry, dim_entry, dim_fixed_enc, wemb):
        self.len_var = max_len_var_enc
        self.dim_var = dim_var
        self.len_entry = max_len_entry
        self.dim_entry = dim_entry
        self.dim_fixed = dim_fixed_enc
        self.dim_interaction = 100
        self.dict_embedding = tf.Variable(wemb, trainable=False)

    # entry : list of voca indices  [N, M]
    # context_var : [N, K, EMB_SIZE]
    # context_fixed : [N, SIZE2]
    def read_entry(self, entry, context_var, dropout):
        def attention_encode_var(input, context_var):
            return attention(input, context_var, self.dim_entry, self.dim_var, self.dim_interaction,
                             "reference_var_context") # [N, len_entry, dim_var]

        def reference_fixed_n_encode(input, input_len, context_fixed):
            cf_info = tf.reshape(tf.tile(context_fixed, input_len),  tf.shape(input)[:2] + tf.shape(context_fixed[2:]))
            x = tf.concat([input, cf_info], axis=2)
            return highway_layer(x, self.dim_fixed+self.dim_entry, tf.nn.relu, "reference_fixed")


        entry_emb = tf.nn.embedding_lookup(self.dict_embedding, entry, name="dict_embedded")  # [batch, max_seq, dim]

        entry_emb_attn = attention_encode_var(entry_emb, context_var)
        #entry_emb_enco = reference_fixed_n_encode(entry_emb, self.len_entry, context_fixed)

        # This is variable length dictionary entry representation
        var_entry = tf.concat([entry_emb, entry_emb_attn], axis=1)
        var_entry_dim = self.dim_entry + self.dim_var # + (self.dim_entry+self.dim_fixed)
        fixed_entry = LSTM_pool(var_entry, self.len_entry, var_entry_dim, dropout, "entry_lstm")
        return fixed_entry

