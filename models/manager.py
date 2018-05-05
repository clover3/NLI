from tensorflow.python.client import device_lib
import os
from models.CAFE import *
from util import *
from collections import Counter
from random import shuffle

LOCAL_DEVICES = device_lib.list_local_devices()
from tensorflow.python.client import timeline
from deepexplain.tensorflow import DeepExplain
from models import adverserial
from models.common import get_batches, load_pickle, load_wemb


def get_summary_path(name):
    i = 0

    def gen_path():
        return os.path.join('summary', '{}{}'.format(name, i))

    while os.path.exists(gen_path()):
        i += 1

    return gen_path()


class Manager:
    def __init__(self, max_sequence, word_indice, batch_size, num_classes,
                 vocab_size, embedding_size, lstm_dim):
        print("Building Model")
        print("Batch size : {}".format(batch_size))
        print("LSTM dimension: {}".format(lstm_dim))
        self.premise_x = tf.placeholder(
            tf.int32, [None, max_sequence], name='premise')
        self.hypothesis_x = tf.placeholder(
            tf.int32, [None, max_sequence], name='hypothesis')
        self.input_h = tf.placeholder(
            tf.int32, [None, max_sequence], name="input_h_absolute_input")
        self.premise_pos = tf.placeholder(
            tf.int32, [None, max_sequence, 47], name='premise_pos')
        self.hypothesis_pos = tf.placeholder(
            tf.int32, [None, max_sequence, 47], name='hypothesis_pos')
        self.premise_char = tf.placeholder(
            tf.int32, [None, max_sequence, args.char_in_word_size],
            name='premise_char')
        self.hypothesis_char = tf.placeholder(
            tf.int32, [None, max_sequence, args.char_in_word_size],
            name='hypothesis_char')
        self.premise_exact_match = tf.placeholder(
            tf.int32, [None, max_sequence, 1], name='premise_exact_match')
        self.hypothesis_exact_match = tf.placeholder(
            tf.int32, [None, max_sequence, 1], name='hypothesis_exact_match')

        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob_absolute_input")
        self.batch_size = batch_size
        self.filters = []

        self.num_classes = num_classes
        self.hidden_size = 200
        self.vocab_size = vocab_size
        self.sent_crop_len = 100

        self.embedding_size = embedding_size
        self.max_seq = max_sequence
        self.lstm_dim = lstm_dim
        self.reg_constant = 1e-5
        self.lr = 3e-4

        self.W = None
        self.word_indice = word_indice

        self.l2_loss = 0

        self.train_op = None
        self.emb_train = False

        with tf.name_scope("embedding"):
                self.embedding = tf.Variable(load_wemb(self.word_indice,
                                                       self.embedding_size),
                                             trainable=False)

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)

        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.device('/gpu:0'):
            self.network()
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)
            self.train_op = self.get_train_op()
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    def log_info(self):
        self.run_metadata = tf.RunMetadata()
        path = get_summary_path("train")
        train_log_path = os.path.join(path, "train")
        test_log_path = os.path.join(path, "test")
        self.train_writer = tf.summary.FileWriter(train_log_path,
                                                  self.sess.graph)
        self.train_writer.add_run_metadata(self.run_metadata, "train")
        print("Summary at {}".format(path))
        self.test_writer = tf.summary.FileWriter(test_log_path,
                                                 filename_suffix=".test")

    def get_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        return optimizer.apply_gradients(grads_and_vars,
                                         global_step=self.global_step)

    def network(self):
        with DeepExplain(session=self.sess, graph=self.sess.graph) as de:
            with tf.name_scope("embedding"):
                if not os.path.exists("pickle/wemb"):
                    self.embedding = load_embedding(self.word_indice,
                                                    self.embedding_size)
                else:
                    self.embedding = tf.Variable(load_pickle("wemb"),
                                                 trainable=False)

            logits = cafe_network(self.premise_x, self.hypothesis_x,
                                  self.premise_pos, self.hypothesis_pos,
                                  self.premise_exact_match,
                                  self.hypothesis_exact_match,
                                  self.premise_char, self.hypothesis_char,
                                  self.batch_size, self.num_classes,
                                  self.embedding, self.embedding_size,
                                  self.max_seq, self.l2_loss,
                                  self.dropout_keep_prob, self.is_train)

            self.logits = tf.identity(logits, name="absolute_output")
            print(self.input_y)
            pred_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.input_y, logits=self.logits))

            self.acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.logits, axis=1),
                                 tf.cast(self.input_y, tf.int64)), tf.float32))

            l2_loss = tf.losses.get_regularization_loss()
            self.loss = pred_loss + self.reg_constant * l2_loss

        tf.summary.scalar('l2loss', self.reg_constant * l2_loss)
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.loss)

        return self.loss

    def load(self, id):
        self.saver.restore(self.sess, self.load_path(id))

    def save_path(self):
        directory = os.path.join(os.path.abspath("checkpoint"), "hdrop",
                                 "model")
        if not os.path.isdir(directory):
            os.makedirs(directory)

        return directory

    def load_path(self, id):
        return os.path.join(os.path.abspath("checkpoint"), id)

    def view_weights(self, dev_data):
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")

        def run_result(batch):
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches
            return self.sess.run([self.logits, feature], feed_dict={
                self.premise_x: p,
                self.hypothesis_x: h,
                self.premise_pos: p_pos,
                self.hypothesis_pos: h_pos,
                self.premise_char: p_char,
                self.hypothesis_char: h_char,
                self.premise_exact_match: p_exact,
                self.hypothesis_exact_match: h_exact,
                self.input_y: y,
                self.dropout_keep_prob: 1.0
            })

        batch_size = 30
        dev_batches = get_batches(dev_data, batch_size, 100)
        run_logits, run_feature = run_result(dev_batches[0])

        _, dim4 = feature.get_shape().as_list()
        dim = int(dim4 / 4)

        D = {0: "E",
             1: "N",
             2: "C"}
        p, p_len, h, h_len, y = dev_batches[0]
        for i in range(batch_size):
            print("-------")
            pred = np.argmax(run_logits, axis=1)

            true_label = D[y[i]]
            pred_label = D[pred[i]]
            print("--- {}({}) -- {} --- ".format(pred_label, true_label,
                                                 run_logits[i]))
            prem = run_feature[i, 0:dim*1]
            hypo = run_feature[i, dim * 1:dim*2]
            sub = run_feature[i, dim*2:dim*3]
            print("concat*sub/|hypo|:", np.dot(prem, sub)/np.dot(hypo, hypo))
            print("Concat:", end="")
            for j in range(dim*2):
                print("{0:.1f} ".format(run_feature[i, j]), end="")
            print()
            print("sub:", end="")
            for j in range(dim):
                print("{0:.1f} ".format(run_feature[i, dim*2+j]), end="")
            print()
            print("odot:", end="")
            for j in range(dim):
                print("{0:.1f} ".format(run_feature[i, dim*3+j]), end="")
            print()

    def view_weights2(self, dev_data):
        pred_high1_w = tf.get_default_graph().get_tensor_by_name(
            name="pred/high1/weight:0")
        pred_high2_w = tf.get_default_graph().get_tensor_by_name(
            name="pred/high2/weight:0")
        pred_dense_w = tf.get_default_graph().get_tensor_by_name(
            name="pred/dense/W:0")
        pred_dense_b = tf.get_default_graph().get_tensor_by_name(
            name="pred/dense/b:0")

        def run_result(batch):
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches
            return self.sess.run(
                [self.logits, pred_high1_w, pred_high2_w,
                 pred_dense_w, pred_dense_b], feed_dict={
                     self.premise_x: p,
                     self.hypothesis_x: h,
                     self.premise_pos: p_pos,
                     self.hypothesis_pos: h_pos,
                     self.premise_char: p_char,
                     self.hypothesis_char: h_char,
                     self.premise_exact_match: p_exact,
                     self.hypothesis_exact_match: h_exact,
                     self.input_y: y,
                     self.dropout_keep_prob: 1.0
                     })

        batch_size = 30
        dev_batches = get_batches(dev_data, batch_size, 100)
        (run_logits, pred_high1_w_out, pred_high2_w,
         pred_dense_w, pred_dense_b) = run_result(dev_batches[0])
        print(pred_high1_w_out)

    def lrp_entangle(self, dev_data, idx2word):
        batch_size = 30
        max_seq = 100
        D = {0: "E",
             1: "N",
             2: "C"}

        dev_batches = get_batches(dev_data, batch_size, max_seq)

        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"

        def run_result(batch):
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches
            logits, = self.sess.run([self.logits], feed_dict={
                self.premise_x: p,
                self.hypothesis_x: h,
                self.premise_pos: p_pos,
                self.hypothesis_pos: h_pos,
                self.premise_char: p_char,
                self.hypothesis_char: h_char,
                self.premise_exact_match: p_exact,
                self.hypothesis_exact_match: h_exact,
                self.input_y: y,
                self.dropout_keep_prob: 1.0
            })
            return logits

        ENTAILMENT = 0
        PREMISE = 0
        HYPOTHESIS = 1

        p, p_len, h, h_len, y = dev_batches[0]
        run_logits = run_result(dev_batches[0])
        print_shape("p", p)
        print_shape("p_len", p_len)
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        _, f_size = feature.get_shape().as_list()
        f_size = int(f_size/2)
        # f_size = 100 # debug
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(
            name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(
            name="hypothesis:0")
        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h,
                       self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            stop = [p_emb_tensor, h_emb_tensor]
            f_begin = args.frange

            def evaluate_E():
                E_list = []
                for f in range(f_begin, f_begin+100):
                    begin = time.time()
                    raw_E = de.explain('grad*input', feature[:, f], stop,
                                       x_input, xi)
                    raw_E[PREMISE] = np.sum(raw_E[PREMISE], axis=2)
                    raw_E[HYPOTHESIS] = np.sum(raw_E[HYPOTHESIS], axis=2)
                    E_list.append(raw_E)
                    print("Elapsed={}".format(time.time() - begin))
                return E_list

            # E_list = evaluate_E()
            # save_pickle("f_{}".format(f_begin), E_list)

            # save_pickle("E_list", E_list)

            def load_E_particle():
                E_list_list = load_pickle("f_s")
                result = []
                for mini in E_list_list:
                    for elem in mini:
                        result.append(elem)
                return result

            E_list = load_E_particle()

            print("f_size : {}".format(f_size))
            print("E[0][0].shape: {}".format(E_list[0][PREMISE].shape))

            for b in range(batch_size):
                print("-------")
                pred = np.argmax(run_logits, axis=1)

                true_label = D[y[b]]
                pred_label = D[pred[b]]

                print("--- {}({}) -- {} --- ".format(pred_label, true_label,
                                                     run_logits[b]))

                entangle = np.zeros([p_len[b], h_len[b]])
                entangle_p = np.zeros([p_len[b], p_len[b]])

                r_i_sum = [np.zeros(p_len[b]), np.zeros(h_len[b])]
                s_len = [p_len[b], h_len[b]]
                for s in [PREMISE, HYPOTHESIS]:
                    for i in range(s_len[s]):
                        for f in range(f_size):
                            r_i_sum[s][i] += abs(E_list[f][s][b, i])

                for f in range(f_size):
                    p_r = E_list[f][PREMISE]
                    h_r = E_list[f][HYPOTHESIS]

                    for i_p in range(p_len[b]):
                        for i_h in range(h_len[b]):
                            entangle[i_p, i_h] += abs(p_r[b, i_p]) * abs(
                                h_r[b, i_h])
                    for i1 in range(p_len[b]):
                        for i2 in range(p_len[b]):
                            entangle_p[i1, i2] += abs(p_r[b, i1]) * abs(
                                p_r[b, i2])
                print("Intra")
                for i1 in range(p_len[b]):
                    for i2 in range(p_len[b]):
                        print("{0:.0f}\t".format(100*entangle_p[i1, i2]),
                              end="")
                    print("")
                print("Inter")

                for i_p in range(p_len[b]):
                    print("{}){}".format(i_p, word(p[b, i_p])), end=" ")
                print("")
                print("\t", end="")
                for i_h in range(h_len[b]):
                    print("{}){}".format(i_h, word(h[b, i_h])), end=" ")
                print("")

                for i_p in range(p_len[b]):
                    print("{}:\t".format(i_p), end="")
                    for i_h in range(h_len[b]):
                        print("{0:.2f}\t".format(100*entangle[i_p, i_h]),
                              end="")
                    print("")
                print("Marginal ")
                entangle_m_p = np.sum(entangle, axis=1)
                entangle_m_h = np.sum(entangle, axis=0)
                print("< premise >")
                for i_p in range(p_len[b]):
                    print("{0:.0f}\t{1}".format(entangle_m_p[i_p],
                                                word(p[b, i_p])))
                print("< hypothesis >")
                for i_h in range(h_len[b]):
                    print("{0:.0f}\t{1}".format(entangle_m_h[i_h],
                                                word(h[b, i_h])))

    def lrp_3way(self, dev_data, idx2word):

        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"

        for v in tf.global_variables():
            print(v)

        soft_out = tf.nn.softmax(self.logits)

        def run_result(batch):
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches
            logits, = self.sess.run([soft_out], feed_dict={
                self.premise_x: p,
                self.hypothesis_x: h,
                self.premise_pos: p_pos,
                self.hypothesis_pos: h_pos,
                self.premise_char: p_char,
                self.hypothesis_char: h_char,
                self.premise_exact_match: p_exact,
                self.hypothesis_exact_match: h_exact,
                self.input_y: y,
                self.dropout_keep_prob: 1.0
            })
            return logits

        D = {0: "E",
             1: "N",
             2: "C"}

        print("view lrp")
        # Print highway1
        # Print Highway2
        # Print pred/dense
        batch_size = 30
        dev_batches = get_batches(dev_data, batch_size, 100)
        p, p_len, h, h_len, y = dev_batches[0]
        run_logits = run_result(dev_batches[0])
        print_shape("p", p)
        print_shape("p_len", p_len)
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        print_shape("feature", feature)

        _, dim4 = feature.get_shape().as_list()
        dim = int(dim4 / 4)
        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h,
                       self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            stop = [feature]
            E_all = []
            for label in range(3):
                E_all.append(de.explain('grad*input', soft_out[:, label], stop,
                                        x_input, xi))
            for i in range(batch_size):
                print("-------")
                pred = np.argmax(run_logits, axis=1)

                true_label = D[y[i]]
                pred_label = D[pred[i]]

                print("--- {}({}) -- {} --- ".format(pred_label,
                                                     true_label, run_logits[i])
                      )
                for label in range(3):
                    r = E_all[label][0]
                    r_concat = np.sum(r[i, 0:dim*2])
                    r_sub = np.sum(r[i, dim*2:dim*3])
                    r_odot = np.sum(r[i, dim*3:dim*4])
                    print(D[label])
                    print("concat {0:.2f} ".format(r_concat))
                    for j in range(0, 200):
                        print("{0:.2f}".format(r[i, j]*100), end=" ")
                    print()
                    print("sub {0:.2f} ".format(r_sub))
                    for j in range(dim*2, dim*2+200):
                        print("{0:.2f}".format(r[i, j]*100), end=" ")
                    print()
                    print("odot {0:.2f} ".format(r_odot))
                    for j in range(dim * 3, dim * 3 + 200):
                        print("{0:.2f}".format(r[i, j] * 100), end=" ")
                    print()

                    # for j in range(dim):
                    #    print("{0:.2f}".format(r[i,j]), end=" ")
                    # print("")

    def manaual_test(self, word2idx, idx2word):
        h1 = ("It 's an interesting account of the violent history of modern "
              "Israel , and ends in the  "
              "Room where nine Jews were executed . ")

    def run_adverserial(self, word2idx):
        test_cases, tag = adverserial.antonym()
        OOV = 0
        PADDING = 1
        max_sequence = 400

        data = []
        for test_case in test_cases:
            p, h, y = test_case
            p, p_len = convert_tokens(p, word2idx)
            h, h_len = convert_tokens(h, word2idx)
            data.append({
                'p': p,
                'p_len': p_len,
                'h': h,
                'h_len': h_len,
                'y': y})

        batches = get_batches(data, 100, 100)

        cate_suc = Counter()
        cate_total = Counter()
        for batch in batches:
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches
            logits, acc = self.sess.run([self.logits, self.acc], feed_dict={
                self.premise_x: p,
                self.hypothesis_x: h,
                self.premise_pos: p_pos,
                self.hypothesis_pos: h_pos,
                self.premise_char: p_char,
                self.hypothesis_char: h_char,
                self.premise_exact_match: p_exact,
                self.hypothesis_exact_match: h_exact,
                self.input_y: y,
                self.dropout_keep_prob: 1.0
            })
            print("Acc : {}".format(acc))
            for i, logit in enumerate(logits):
                # print(" ".join(test_cases[i][0]))
                print("----------")
                print(i)
                print(" ".join(test_cases[i][1]))
                print("y = {}({})".format(np.argmax(logit), test_cases[i][2]))
                print(logit)
                cate_total[tag[i]] += 1
                if np.argmax(logit) == test_cases[i][2]:
                    cate_suc[tag[i]] += 1

        for key in cate_total.keys():
            total = cate_total[key]
            if key in cate_suc:
                suc = cate_suc[key]
            else:
                suc = 0
            print("{}:{}/{}".format(key, suc, total))

    def view_lrp(self, dev_data, idx2word):
        def expand_y(y):
            r = []
            for yi in y:
                if yi == 0:
                    yp = [1, 0, 0]
                elif yi == 1:
                    yp = [0, 1, 0]
                else:
                    yp = [0, 0, 1]

                r.append(yp)
            return np.array(r)

        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"

        for v in tf.global_variables():
            print(v)

        def run_result(batch):
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches
            logits, = self.sess.run([self.logits], feed_dict={
                self.premise_x: p,
                self.hypothesis_x: h,
                self.premise_pos: p_pos,
                self.hypothesis_pos: h_pos,
                self.premise_char: p_char,
                self.hypothesis_char: h_char,
                self.premise_exact_match: p_exact,
                self.hypothesis_exact_match: h_exact,
                self.input_y: y,
                self.dropout_keep_prob: 1.0
                })
            return logits

        D = {0: "E",
             1: "N",
             2: "C"}

        print("view lrp")
        dev_batches = get_batches(dev_data, 100, 100)
        p, p_len, h, h_len, y = dev_batches[0]
        run_logits = run_result(dev_batches[0])
        print_shape("p", p)
        print_shape("p_len", p_len)

        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(
            name="premise:0")

        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(
            name="hypothesis:0")

        def print_color_html(word, r0, r1, r2, r_max, r_min):
            def normalize(val):
                v = (val - r_min) / (r_max - r_min) * 255
                assert(v < 256 and v >= 0)
                return v
            normal_val = [normalize(r) for r in [r0, r1, r2]]
            bg_color = "".join(["%02x" % v for v in normal_val])
            if sum(normal_val) > 256 * 3 * 0.7:
                text_color = "000000"  # black
            else:
                text_color = "ffffff"  # white
            html = ("<span style=\"color:#{}; background-color:"
                    "#{}\">{}</span>&nbsp;\n").format(text_color,
                                                      bg_color, word)
            return html

        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h,
                       self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            yi = expand_y(y)
            stop = [p_emb_tensor, h_emb_tensor]

            c_e = self.logits[:, 2] - self.logits[:, 0]
            e_n = self.logits[:, 0] - self.logits[:, 1]
            C_E = de.explain('elrp', c_e,
                             stop, x_input, xi)
            E_N = de.explain('elrp', e_n, stop, x_input, xi)

            E_all = []
            for label in range(3):
                E_all.append(de.explain('grad*input', self.logits[:, label],
                                        stop, x_input, xi))

            print("result----------")
            pred = np.argmax(run_logits, axis=1)
            f = open("result.html", "w")
            for i in range(100):
                print("-------")
                true_label = D[y[i]]
                pred_label = D[pred[i]]
                # if pred[i] == 2:
                #    E = C_E
                # else:
                #    E = E_N
                E_sum = list(
                    [[np.sum(E_all[label][s][i, :, :],
                             axis=1) for s in range(2)] for label in range(3)])
                r_max = max(
                    [np.max(
                        E_sum[label][s]
                        ) for label in range(3) for s in range(2)])
                r_min = min(
                    [np.min(
                        E_sum[label][s]
                        ) for label in range(3) for s in range(2)])

                p_r = E_all[2][0] - E_all[0][0]
                h_r = E_all[2][1] - E_all[0][1]
                print("--- {}({}) -- {} --- ".format(pred_label,
                                                     true_label,
                                                     run_logits[i]))
                # print("sum[r]={} max={} min={}".format(
                #    np.sum(p_r[i])+ np.sum(h_r[i]), r_max, r_min))
                _, max_seq, _ = p_r.shape
                p_r_s = np.sum(p_r[i], axis=1)
                h_r_s = np.sum(h_r[i], axis=1)

                f.write("<html>")
                f.write(
                    "<div><span>Prediction={} , Truth={}</span><br>\n".format(
                        pred_label, true_label))
                f.write("<p>Premise</p>\n")
                print("")
                print("premise: ")
                r_max = max(
                    [np.max(E_sum[label][s]
                            ) for label in range(3) for s in range(0, 1)])
                r_min = min(
                    [np.min(E_sum[label][s]
                            ) for label in range(3) for s in range(0, 1)])

                for j in range(max_seq):
                    print("{0}({1:.2f})".format(word(p[i, j]),
                                                p_r_s[j]), end=" ")
                    f.write(print_color_html(word(p[i, j]), E_sum[0][0][j],
                                             E_sum[1][0][j], E_sum[2][0][j],
                                             r_max, r_min))

                print()
                _, max_seq, _ = h_r.shape
                f.write("<br><p>Hypothesis</p>\n")
                print("hypothesis: ")
                r_max = max(
                    [np.max(
                        E_sum[label][s]
                        ) for label in range(3) for s in range(1, 2)])
                r_min = min(
                    [np.min(
                        E_sum[label][s]
                        ) for label in range(3) for s in range(1, 2)])
                for j in range(max_seq):
                    print("{0}({1:.2f})".format(word(h[i, j]), h_r_s[j]),
                          end=" ")
                    f.write(print_color_html(word(h[i, j]),
                                             E_sum[0][1][j],
                                             E_sum[1][1][j],
                                             E_sum[2][1][j], r_max, r_min))

                print()
                f.write("</div><hr>")
            f.write("</html>")

    def print_time(self):
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

    def check_dev(self, data, g_step):
        acc_sum = []
        loss_sum = []
        step = 0

        step_per_batch = int(len(data) / 200)
        for j in range(step_per_batch):
            batches = get_batches(data, j*self.batch_size,
                                  (j+1)*self.batch_size, self.sent_crop_len)
            (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
             p_exact, h_exact, y) = batches

            acc, loss, summary = self.sess.run(
                [self.acc, self.loss, self.merged], feed_dict={
                    self.premise_x: p,
                    self.hypothesis_x: h,
                    self.premise_pos: p_pos,
                    self.hypothesis_pos: h_pos,
                    self.premise_char: p_char,
                    self.hypothesis_char: h_char,
                    self.premise_exact_match: p_exact,
                    self.hypothesis_exact_match: h_exact,
                    self.input_y: y,
                    self.dropout_keep_prob: 0.8,
                    self.is_train: True,
                }, run_metadata=self.run_metadata)

            acc_sum.append(acc)
            loss_sum.append(loss)
            self.test_writer.add_summary(summary, g_step+step)
            step += 1

        print("Dev acc={} loss={} ".format(avg(acc_sum), avg(loss_sum)))

    def train(self, epochs, data, valid_data, rerun=False):
        print("Train")
        self.log_info()
        if not rerun:
            self.sess.run(tf.global_variables_initializer())
        # batches = get_batches(data, self.batch_size, self.sent_crop_len)
        # dev_batches = get_batches_val(valid_data, 200, self.sent_crop_len)
        step_per_batch = int(len(data) / self.batch_size)
        log_every = int(step_per_batch/200)
        check_dev_every = int(step_per_batch/5)
        g_step = 0

        for i in range(epochs):
            print("Epoch {}".format(i))
            s_loss = 0
            l_acc = []
            time_estimator = TimeEstimator(self.batch_size, name="epoch")
            # shuffle(batches)

            for j in range(step_per_batch):
                batches = get_batches(data, j*self.batch_size,
                                      (j+1)*self.batch_size,
                                      self.sent_crop_len)
                (p, h, input_p_len, input_h_len, p_pos, h_pos, p_char, h_char,
                 p_exact, h_exact, y) = batches

                g_step += 1

                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)

                _, acc, loss, summary = self.sess.run(
                    [self.train_op, self.acc, self.loss, self.merged],
                    feed_dict={
                        self.premise_x: p,
                        self.hypothesis_x: h,
                        self.premise_pos: p_pos,
                        self.hypothesis_pos: h_pos,
                        self.premise_char: p_char,
                        self.hypothesis_char: h_char,
                        self.premise_exact_match: p_exact,
                        self.hypothesis_exact_match: h_exact,
                        self.input_y: y,
                        self.dropout_keep_prob: 0.8,
                        self.is_train: True,
                        }, run_metadata=self.run_metadata, options=run_options)

                if g_step % log_every == 1 or g_step < 5:
                    print("step{} : {} acc : {} ".format(g_step, loss, acc))
                if g_step % check_dev_every == 0:
                    self.check_dev(valid_data, g_step)
                s_loss += loss
                l_acc.append(acc)
                self.train_writer.add_summary(summary, g_step)
                self.train_writer.add_run_metadata(self.run_metadata,
                                                   "meta_{}".format(g_step))
                time_estimator.tick()

            current_step = tf.train.global_step(self.sess, self.global_step)
            path = self.saver.save(self.sess, self.save_path(),
                                   global_step=current_step)
            print("Checkpoint saved at {}".format(path))
            print("Training Average loss : {} , acc : {}".format(s_loss,
                                                                 avg(l_acc)))
