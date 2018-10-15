from tensorflow.python.client import device_lib

from scipy.stats import pearsonr
from models.CAFE import *
from util import *
from collections import Counter

LOCAL_DEVICES = device_lib.list_local_devices()
from tensorflow.python.client import timeline
from deepexplain.tensorflow import DeepExplain
from models import adverserial
from random import shuffle
from models.entangle import *
import math
from scipy.stats import chisquare
import re

def get_summary_path(name):
    i = 0
    def gen_path():
        return os.path.join('summary', '{}{}'.format(name, i))

    while os.path.exists(gen_path()):
        i += 1

    return gen_path()

def get_run_id():
    i = 0
    def gen_path():
        return os.path.join('summary', '{}{}'.format("train", i))

    while os.path.exists(gen_path()):
        i += 1
    return i-1


class Manager:
    def __init__(self, max_sequence, word_indice, batch_size, num_classes, vocab_size, embedding_size, lstm_dim):
        print("Building Model")
        print("Batch size : {}".format(batch_size))
        print("LSTM dimension: {}".format(lstm_dim))
        self.input_p = tf.placeholder(tf.int32, [None, max_sequence], name="input_p_absolute_input")
        self.input_p_len = tf.placeholder(tf.int64, [None,], name="input_p_len_absolute_input")
        self.input_h = tf.placeholder(tf.int32, [None, max_sequence], name="input_h_absolute_input")
        self.input_h_len = tf.placeholder(tf.int64, [None,], name="input_h_len_absolute_input")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob_absolute_input")
        self.batch_size = batch_size
        self.filters = []

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.sent_crop_len = 100

        self.embedding_size = embedding_size
        self.max_seq = max_sequence
        self.lstm_dim = lstm_dim
        self.reg_constant = 3e-5
        self.lr = args.learning_rate
        print("learning rate : {}".format(self.lr))
        print("Reg lambda = {}".format(self.reg_constant))
        print("FM size= {}".format(args.fm_latent))

        self.W = None
        self.word_indice = word_indice

        self.train_op = None
        config = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.device('/gpu:0'):
            tf.set_random_seed(2)
            self.network()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.get_train_op()
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    def log_info(self):
        self.run_metadata = tf.RunMetadata()
        path = get_summary_path("train")
        train_log_path = os.path.join(path, "train")
        test_log_path = os.path.join(path, "test")
        self.train_writer = tf.summary.FileWriter(train_log_path, self.sess.graph)
        self.train_writer.add_run_metadata(self.run_metadata, "train")
        print("Summary at {}".format(path))
        self.test_writer = tf.summary.FileWriter(test_log_path, filename_suffix=".test")

    def get_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        return optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def network(self):
        with DeepExplain(session=self.sess, graph=self.sess.graph) as de:
            with tf.name_scope("embedding"):
                if not os.path.exists("pickle/wemb"):
                    wemb = load_embedding(self.word_indice, self.embedding_size)

                else:
                    wemb =load_pickle("wemb")

                self.embedding = tf.Variable(wemb, trainable=False)

            logits = cafe_network (self.input_p,
                                   self.input_h,
                                   self.input_p_len,
                                   self.input_h_len,
                                   self.batch_size,
                                   self.num_classes,
                                   self.embedding,
                                   self.embedding_size,
                                   self.max_seq,
                                   self.dropout_keep_prob
                                   )
            self.logits = tf.identity(logits, name="absolute_output")
            pred_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))
            pred_loss = tf.clip_by_value(pred_loss, 1e-10, 100.0)
            self.acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.logits, axis=1),tf.cast(self.input_y, tf.int64)), tf.float32))
            l2_loss = tf.losses.get_regularization_loss()
            self.l2_loss = self.reg_constant * l2_loss
            self.loss = pred_loss + self.reg_constant * l2_loss

        tf.summary.scalar('l2loss', self.reg_constant * l2_loss)
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.loss)

        return self.loss

    def load(self, id):
        path = self.load_path(id)
        print("Loading {}".format(path))
        self.saver.restore(self.sess, path)

    def save_path(self):
        run_name = "{}".format(get_run_id())
        return os.path.join(os.path.abspath("checkpoint"), run_name, "model")

    def load_path(self, id):
        return os.path.join(os.path.abspath("checkpoint"), id)

    def corr_analysis(self, dev_data, idx2word):
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        _, dim4 = feature.get_shape().as_list()
        dim = int(dim4 / 4)

        D = {0: "E",
             1: "N",
             2: "C"}
        batch_size = 30
        dev_batches = get_batches(dev_data, batch_size, 100)
        results = [self.run_w_feature(batch) for batch in dev_batches]
        acc_logits, acc_feature = zip(*results)
        acc_logits = np.concatenate(acc_logits, axis=0)
        acc_feature = np.concatenate(acc_feature, axis=0)

        def one_hot(batch):
            p, p_len, h, h_len, y = batch

            fv_list = []
            for i in range(len(p_len)):
                prem_tokens = set([idx2word[idx] for idx in p[i,:]])
                hypo_tokens = set([idx2word[idx] for idx in h[i,:]])
                fv = np.zeros([15])
                for w_no, word in enumerate(["no", "nothing", "never", "not", "n't"]):
                    if word in prem_tokens:
                        fv[w_no*3] = 1
                    if word in hypo_tokens:
                        fv[w_no*3+1] = 1
                    if word in prem_tokens or word in hypo_tokens:
                        fv[w_no*3+2] = 1
                fv_list.append(fv)
            return np.stack(fv_list)

        acc_fv = np.concatenate([one_hot(batch) for batch in dev_batches], axis=0)

        def chi_test(ce_logits, feature):
            logit_p = (ce_logits > 0.4)
            logit_n = np.logical_not(logit_p)
            prob_logit_p = np.average(logit_p)


            feature_avg = np.average(feature)
            feature_p = (feature > feature_avg)
            feature_n = np.logical_not(feature_p)
            prob_feature_p = np.average(feature_p)

            total = len(feature)
            pp_expected = prob_logit_p * prob_feature_p * total
            pn_expected = prob_logit_p * (1-prob_feature_p) * total
            np_expected = (1-prob_logit_p) * prob_feature_p * total
            nn_expected = (1-prob_logit_p) * (1-prob_feature_p) * total

            pp_actual = np.logical_and(logit_p, feature_p).sum()
            pn_actual = np.logical_and(logit_p, feature_n).sum()
            np_actual = np.logical_and(logit_n, feature_p).sum()
            nn_actual = np.logical_and(logit_n, feature_n).sum()
            actual_l = [pp_actual, pn_actual, np_actual, nn_actual]
            expected_l = [pp_expected, pn_expected, np_expected, nn_expected]
            x2, p = chisquare(actual_l,
                          expected_l,
                          ddof=2)
            if x2 > 3000:
                print(actual_l, expected_l)

            return x2, feature_avg, pp_actual>pp_expected


        corr = []
        for label in [0,1,2]:
            raw = [(f, pearsonr(acc_logits[:, label], acc_feature[:, f])) for f in range(dim4)]
            corr.append(sorted(raw, key=lambda x: x[1][0]))

        stdev = np.std(acc_feature, axis=0)
        corr_ce = [(f, pearsonr(acc_logits[:, 2]-acc_logits[:,0], acc_feature[:, f])) for f in range(dim4)]
        corr_ce.sort(key=lambda x: x[1][0])
        corr_entail = corr[0]
        corr_neutra = corr[1]
        corr_contra = corr[2]
        chi_ce = [(f, chi_test(acc_logits[:, 2]-acc_logits[:,0], acc_feature[:, f])) for f in range(dim4)]
        chi_ce.sort(key=lambda x: x[1][0], reverse=True)
        print("Chi square test")
        for f_id, info in chi_ce[:30]:
            x2, cut, slope = info
            print("{} :{}".format(f_id, info))
        save_pickle("chi_ce", chi_ce[:30])
        def has_signal(feature_v):
            for f_id, info in chi_ce[:100]:
                x2, cut, slope = info

                if slope and feature_v[f_id] > cut:
                    return True
                if not slope and feature_v[f_id] < cut:
                    return True
            return False

        count_shift = 0
        count_ex = 0
        for i in range(len(acc_feature)):
            p = acc_logits[i, 2] - acc_logits[i, 0] > 0.4
            if not has_signal(acc_feature[i]):
                count_ex += 1
                if p:
                    count_shift += 1

        print("{} / {} ".format(count_shift, count_ex))


        return
        c_printed = []
        print("contra - major feature")
        interest = range(dim4)

        for index in interest:
            print(corr_contra[index])
            c_printed.append(corr_contra[index][0])

        e_printed = []
        print("entail - major feature")
        for index in interest:
            print(corr_entail[index])
            e_printed.append(corr_entail[index][0])

        print("neutral - major feature")
        for index in interest:
            print(corr_neutra[index])

        intersection = set(c_printed) & set(e_printed)
        print("c,e common ")
        for entry in corr_ce:
            if entry[0] in intersection:
                print(entry)

        def region(f):
            if f < dim:
                return "premise"
            elif f < dim*2:
                return "hypothesis"
            elif f < dim*3:
                return "sub"
            else:
                return "odot"

        threshold = 1e-8
        counter = Counter()
        major_ce = []
        for i in range(dim4):
            entry = corr_ce[i]
            f_id = entry[0]
            p_value = entry[1][1]
            n_slope = entry[1][0] / stdev[f_id]
            if p_value < threshold and abs(n_slope) > 0.4:
                counter[region(f_id)] += 1
                major_ce.append(entry)
                print("fid={} stdev={} raw_slop={}".format(f_id, stdev[f_id], entry[1][0]))

        for key, value in counter.items():
            print("{} : {}".format(key, value))


        corr_f = [((entry[0], fv), pearsonr(acc_feature[:, entry[0]], acc_fv[:, fv])) for entry in major_ce for fv in range(15)]
        save_pickle("major_ce", major_ce)


    def view_weights(self, dev_data):
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        def run_result(batch):
            p, p_len, h, h_len, y = batch
            return self.sess.run([self.logits, feature], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
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
            print("--- {}({}) -- {} --- ".format(pred_label, true_label, run_logits[i]))
            prem = run_feature[i,0:dim*1]
            hypo = run_feature[i, dim * 1:dim*2]
            sub = run_feature[i,dim*2:dim*3]
            print("concat*sub/|hypo|:", np.dot(hypo,sub)/ np.dot(hypo,hypo))
            print("Concat:",end="")
            for j in range(dim*2):
                print("{0:.1f} ".format(run_feature[i,j]), end="")
            print()
            print("sub:", end="")
            for j in range(dim):
                print("{0:.1f} ".format(run_feature[i,dim*2+j]), end="")
            print()
            print("odot:", end="")
            for j in range(dim):
                print("{0:.1f} ".format(run_feature[i,dim*3+j]), end="")
            print()


    def poly_weights(self, dev_data):
        weight = tf.get_default_graph().get_tensor_by_name(name="predict/W:0")

        def run_result(batch):
            p, p_len, h, h_len, y = batch
            return self.sess.run([self.logits, weight], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })

        batch_size = 30
        dev_batches = get_batches(dev_data, batch_size, 100)
        run_logits, run_weight = run_result(dev_batches[0])

        dim_all, n_label = weight.get_shape().as_list()
        print("Dimension : {}".format(dim_all))
        dim = 100
        D = {0: "E",
             1: "N",
             2: "C"}
        p, p_len, h, h_len, y = dev_batches[0]
        for i in range(batch_size):
            print("-------")

            for label in range(3):
                print("Linear:", end="")
                for j in range(dim):
                    print("{0:.1f} ".format(run_weight[j, label]), end="")
                print()
                print("Square:", end="")
                for j in range(dim):
                    print("{0:.1f} ".format(run_weight[dim+j, label]), end="")
                print()
                print("XY:", end="")
                for j in range(1000):
                    print("{0:.1f} ".format(run_weight[dim*2+j, label]), end="")
                print()

    def run_w_feature(self, batch):
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        p, p_len, h, h_len, y = batch
        return self.sess.run([tf.nn.softmax(self.logits), feature], feed_dict={
            self.input_p: p,
            self.input_p_len: p_len,
            self.input_h: h,
            self.input_h_len: h_len,
            self.input_y: y,
            self.dropout_keep_prob: 1.0,
        })

    def run_result(self, batch):
        p, p_len, h, h_len, y = batch
        logits, = self.sess.run([self.logits], feed_dict={
            self.input_p: p,
            self.input_p_len: p_len,
            self.input_h: h,
            self.input_h_len: h_len,
            self.input_y: y,
            self.dropout_keep_prob: 1.0,
        })
        return logits

    @staticmethod
    def word(index, idx2word):
        if index in idx2word:
            if idx2word[index] == "<PADDING>":
                return "PADDING"
            else:
                return idx2word[index]
        else:
            return "OOV"

    def view_weights2(self, dev_data):
        pred_high1_w = tf.get_default_graph().get_tensor_by_name(name="pred/high1/weight:0")
        pred_high2_w = tf.get_default_graph().get_tensor_by_name(name="pred/high2/weight:0")
        pred_dense_w = tf.get_default_graph().get_tensor_by_name(name="pred/dense/W:0")
        pred_dense_b = tf.get_default_graph().get_tensor_by_name(name="pred/dense/b:0")


        def run_result(batch):
            p, p_len, h, h_len, y = batch
            return self.sess.run([self.logits, pred_high1_w, pred_high2_w, pred_dense_w, pred_dense_b], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
        batch_size = 30
        dev_batches = get_batches(dev_data, batch_size, 100)
        run_logits, pred_high1_w_out, pred_high2_w, pred_dense_w, pred_dense_b = run_result(dev_batches[0])
        print(pred_high1_w_out)

    def lrp_premise(self, dev_data, idx2word):
        batch_size = 30
        max_seq = 100
        PREMISE = 0
        HYPOTHESIS = 1
        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"



        dev_batches = get_batches(dev_data, batch_size, max_seq)
        p, p_len, h, h_len, y = dev_batches[0]
        run_logits = self.run_result(dev_batches[0])
        enc_prem = tf.get_default_graph().get_tensor_by_name(name="encode_p_lstm:0")
        _, f_size = enc_prem.get_shape().as_list()
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="hypothesis:0")
        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h, self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            stop = [p_emb_tensor, h_emb_tensor]
            f_begin = args.frange
            def evaluate_E():
                E_list = []
                end = min(f_begin, f_begin+50)
                for f in range(f_begin, end):
                    begin = time.time()
                    raw_E = de.explain('grad*input', enc_prem[:,f], stop, x_input, xi)
                    raw_E[PREMISE] = np.sum(raw_E[PREMISE], axis=2)
                    raw_E[HYPOTHESIS] = np.sum(raw_E[HYPOTHESIS], axis=2)
                    E_list.append(raw_E)
                    print("Elapsed={}".format(time.time() - begin))
                return E_list
            #E_list = evaluate_E()
            #save_pickle("lstm_p_f_{}".format(f_begin), E_list)
            E_list = load_pickle("lstm_p")
            for b in range(batch_size):
                entangle_p = np.zeros([p_len[b], p_len[b]])
                for f in range(f_size):
                    p_r = E_list[f][PREMISE]
                    h_r = E_list[f][HYPOTHESIS]
                    p_r_sum = np.sum(p_r, axis=1)

                    for i1 in range(p_len[b]):
                        for i2 in range(p_len[b]):
                            entangle_p[i1, i2] += abs(p_r[b,i1]) * abs(p_r[b,i2])
                print("Intra")
                for i1 in range(p_len[b]):
                    for i2 in range(p_len[b]):
                        print("{0:.0f}\t".format(100*entangle_p[i1, i2]), end="")
                    print("")
                grouping = minimum_entangle(entangle_p)
                print(grouping.loss_group())
                group = grouping.group_by_count(int(p_len[b]/2))
                for begin, end in group:
                    word_group = []
                    for i in range(begin,end):
                        word_group.append(word(p[b,i]))
                    print("[{}] ".format(" ".join(word_group)), end="")
                print("")

            return

    def feature_invertor(self, dev_data, idx2word):
        batch_size = 30
        max_seq = 100
        PREMISE = 0
        HYPOTHESIS = 1

        D = {0: "E",
             1: "N",
             2: "C"}

        dev_batches = get_batches(dev_data, batch_size, max_seq)
        dim = 612
        major_ce = load_pickle("major_ce")
        c_features = set([entry[0] for entry in major_ce])
        def find_feature(f):
            if f in c_features:
                for entry in chi_ce:
                    if entry[0] == f:
                        return entry[1]
            else:
                return None
        E_list = load_pickle("E_list")
        chi_ce = load_pickle("chi_ce")
        c_features = set([entry[0] for entry in chi_ce])

        p, p_len, h, h_len, y = dev_batches[0]
        run_logits, feature = self.run_w_feature(dev_batches[0])
        pred = np.argmax(run_logits, axis=1)
        feature_ce = load_pickle("feature_ce")

        def word_feature_gradient():
            summary = []
            for b in range(batch_size):
                true_label = D[y[b]]
                pred_label = D[pred[b]]
                print("--- {}({}) -- {} --- ".format(pred_label, true_label, run_logits[b, :]))
                prem_r = np.zeros([p_len[b]])
                hypo_r = np.zeros([h_len[b]])
                for f in range(612*4):
                    if f in c_features:
                        x2, cut, slope = find_feature(f)
                        p_r = E_list[f][PREMISE]
                        h_r = E_list[f][HYPOTHESIS]
                        factor = 0
                        if slope :
                            if feature[b,f] > cut :
                                factor = 1
                        elif not slope:
                            if feature[b,f] < cut:
                                factor = 1
                        for i_p in range(p_len[b]):
                            prem_r[i_p] += abs(p_r[b,i_p]) * factor
                        for i_h in range(h_len[b]):
                            hypo_r[i_h] += abs(h_r[b,i_h]) * factor


                print("premise: " , end = "")
                p_summary =[]
                for i_p in range(p_len[b]):
                    word = self.word(p[b, i_p], idx2word)
                    score= prem_r[i_p]
                    if abs(score) > 0.1:
                        print("{0}({1:.2f})".format(word, score), end=" ")
                    p_summary.append((word, score))
                print("")
                print("Hypo: ", end = "")
                h_summary = []
                for i_h in range(h_len[b]):
                    word = self.word(h[b, i_h], idx2word)
                    score = hypo_r[i_h]
                    print("{0}({1:.2f})".format(word, score), end=" ")
                    h_summary.append((word, score))
                summary.append((h_summary, p_summary))
                print("")
                s1 = np.sum(prem_r)
                s2 = np.sum(hypo_r)
                s3 = s1+s2
                print("Total {}+{}={}".format(s1,s2,s3))
            save_pickle("ce_lrp", summary)
        word_feature_gradient()
        def construct_classification_network(feature):
            high1_W = tf.get_default_graph().get_tensor_by_name("pred/high1/weight:0")
            high1_b = tf.get_default_graph().get_tensor_by_name("pred/high1/bias:0")
            high1_W_T = tf.get_default_graph().get_tensor_by_name("pred/high1/transform_gate/weight:0")
            high1_b_T = tf.get_default_graph().get_tensor_by_name("pred/high1/transform_gate/bias:0")
            high1_param = (high1_W, high1_b, high1_W_T, high1_b_T)

            high2_W = tf.get_default_graph().get_tensor_by_name("pred/high2/weight:0")
            high2_b = tf.get_default_graph().get_tensor_by_name("pred/high2/bias:0")
            high2_W_T = tf.get_default_graph().get_tensor_by_name("pred/high2/transform_gate/weight:0")
            high2_b_T = tf.get_default_graph().get_tensor_by_name("pred/high2/transform_gate/bias:0")
            high2_param = (high2_W, high2_b, high2_W_T, high2_b_T)

            pred_W = tf.get_default_graph().get_tensor_by_name("pred/dense/W:0")
            pred_b = tf.get_default_graph().get_tensor_by_name("pred/dense/b:0")
            pred_param = (pred_W, pred_b)

            high1_param, high2_param, pred_param = self.sess.run([high1_param, high2_param, pred_param])
            logit = feature_classification(feature, high1_param, high2_param, pred_param)
            return logit





    def eval_lrp_feature(self, dev_data):

        batch_size = 30
        max_seq = 100
        D = {0: "E",
             1: "N",
             2: "C"}

        dev_batches = get_batches(dev_data, batch_size, max_seq)
        PREMISE = 0
        HYPOTHESIS = 1

        p, p_len, h, h_len, y = dev_batches[0]
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        _, f_size = feature.get_shape().as_list()
        # f_size = 100 # debug
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="hypothesis:0")

        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h, self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            stop = [p_emb_tensor, h_emb_tensor]
            f_begin = args.frange

            def evaluate_E():
                E_list = []
                for f in range(f_begin, f_begin + 50):
                    begin = time.time()
                    raw_E = de.explain('grad*input', feature[:, f], stop, x_input, xi)
                    raw_E[PREMISE] = np.sum(raw_E[PREMISE], axis=2)
                    raw_E[HYPOTHESIS] = np.sum(raw_E[HYPOTHESIS], axis=2)
                    E_list.append(raw_E)
                    print("Elapsed={}".format(time.time() - begin))
                return E_list

            E_list = evaluate_E()
            save_pickle("temp_f_{}".format(f_begin), E_list)

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
            p, p_len, h, h_len, y = batch
            logits, = self.sess.run([self.logits], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
            return logits

        ENTAILMENT = 0
        PREMISE = 0
        HYPOTHESIS = 1

        p, p_len, h, h_len, y = dev_batches[0]
        run_logits = run_result(dev_batches[0])
        ce_summary = load_pickle("ce_lrp")
        print_shape("p", p)
        print_shape("p_len", p_len)
        feature = tf.get_default_graph().get_tensor_by_name(name="feature:0")
        _, f_size = feature.get_shape().as_list()
        _, feature_out = self.run_w_feature(dev_batches[0])
        #f_size = 100 # debug
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="hypothesis:0")

        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h, self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            stop = [p_emb_tensor, h_emb_tensor]
            f_begin = args.frange
            def evaluate_E():
                E_list = []
                for f in range(f_begin, f_begin+100):
                    begin = time.time()
                    raw_E = de.explain('grad*input', feature[:,f], stop, x_input, xi)
                    raw_E[PREMISE] = np.sum(raw_E[PREMISE], axis=2)
                    raw_E[HYPOTHESIS] = np.sum(raw_E[HYPOTHESIS], axis=2)
                    E_list.append(raw_E)
                    print("Elapsed={}".format(time.time() - begin))
                return E_list

            #E_list = evaluate_E()
            #save_pickle("f_{}".format(f_begin), E_list)

            #save_pickle("E_list", E_list)

            def load_E_particle():
                E_list_list = load_pickle("f_s")
                result = []
                for mini in E_list_list:
                    for elem in mini:
                        result.append(elem)
                return result

            soft_out = tf.nn.softmax(self.logits)
            feature_e = de.explain('grad*input', soft_out[:,0], [feature], x_input, xi)
            feature_c = de.explain('grad*input', soft_out[:,2], [feature], x_input, xi)

            feature_ce = feature_e[0] - feature_c[0]
            save_pickle("feature_ce", feature_ce)

            #E_list = load_E_particle()
            E_list = load_pickle("E_list")
            print("f_size : {}".format(f_size))
            f_begin = 612*2
            f_end = 612*3
            print("E[0][0].shape: {}".format(E_list[0][PREMISE].shape))

            def softmax(w, t=1.0):
                e = np.exp(w / t)
                dist = e / np.sum(e)
                return dist


            def print_table(f, entangle, p_len, h_len, p, h, summary):
                if summary is not None:
                    h_summary, p_summary = summary
                cap = np.max(entangle)
                def gen_color(v):

                    b = int(255- v/cap * 255)
                    b = max(255-v*100,0)
                    return ("%02x" % b) + ("%02x" % b) + "ff"

                def ce_color(v):
                    v= v*50
                    if v> 255:
                        v =255
                    elif v< -255:
                        v=-255
                    b = int(255 - abs(v))
                    if v >= 0 : # red
                        return "ff" + ("%02x" % b) + ("%02x" % b)
                    else:
                        return ("%02x" % b) + "ff" + ("%02x" % b)

                f.write("<table style=\"border:1px solid\">")
                f.write("<tr>")
                f.write("<th></th>")
                for i in range(h_len):
                    if summary is not None:
                        ce_word, ce_val = h_summary[i]
                        f.write("<th style=\"min-width:35px\" bgcolor=\"#{}\">{}</th>".format(ce_color(ce_val), word(h[i])))
                        assert (word(h[i]) == ce_word)
                    else:
                        f.write("<th style=\"min-width:35px\">{}</th>".format(word(h[i])))



                f.write("</tr>")
                for i in range(p_len):
                    if summary is None:
                        f.write("<tr>")
                        f.write("<td>{}</td>".format(word(p[i])))
                    else:
                        ce_word, ce_val = p_summary[i]
                        f.write("<tr>")
                        f.write("<td bgcolor=\"#{}\">{}</td>".format(ce_color(ce_val), word(p[i])))
                    for i2 in range(h_len):
                        f.write("<td bgcolor=\"#{}\"></td>".format(gen_color(entangle[i,i2])))
                    f.write("</tr>")
                f.write("</table>")

            logit_array = []
            pred = np.argmax(run_logits, axis=1)
            html_f = open("Entangles.html", "w")
            html_f.write("<html><body>\n")
            html_f.write("<p><b><font color=\"red\">Red</font><font color=\"green\">/Green</font><b> color in the words implies each words contribution to Contradiction/Entailment</p>\n")
            html_f.write("<p><b>Blue<b> color in the cell represents degree of interaction between two words.</p>\n")
            for b in range(batch_size):
                print("-------")


                true_label = D[y[b]]
                pred_label = D[pred[b]]
                print("--- {}({}) -- {} --- ".format(pred_label, true_label, run_logits[b,:]))
                logit_array.append(run_logits[b,:])

                entangle = np.zeros([p_len[b], h_len[b]])
                entangle_p = np.zeros([p_len[b], p_len[b]])

                r_i_sum = [np.zeros(p_len[b]), np.zeros(h_len[b])]
                s_len = [p_len[b], h_len[b]]
                for s in [PREMISE, HYPOTHESIS]:
                    for i in range(s_len[s]):
                        for f in range(f_size):
                            r_i_sum[s][i] += abs(E_list[f][s][b,i])

                for f in range(f_begin, f_end):
                    p_r = E_list[f][PREMISE]
                    h_r = E_list[f][HYPOTHESIS]
                    p_r_sum = np.sum(p_r, axis=1)
                    r_f = feature_c[0][b, f] - feature_e[0][b, f]
                    #factor = math.exp(r_f)
                    f_i_p = feature_out[b, f-612*2]
                    f_i_h = feature_out[b, f-612]
                    if f_i_p == 0.0 and f_i_h == 0.0:
                        factor = 0
                    else:
                        factor = math.exp(-(abs(0.1 * (f_i_h-f_i_p)/(abs(f_i_h)+abs(f_i_p)))))

                    for i_p in range(p_len[b]):
                        for i_h in range(h_len[b]):
                            entangle[i_p, i_h] += factor * abs(p_r[b,i_p]) * abs(h_r[b,i_h])
                    for i1 in range(p_len[b]):
                           for i2 in range(p_len[b]):
                            entangle_p[i1, i2] += abs(p_r[b,i1]) * abs(p_r[b,i2])

                html_f.write("<h3>Prediction={} True Label={}</h3>".format(pred_label, true_label))
                print_table(html_f, entangle, p_len[b], h_len[b], p[b,:], h[b,:], ce_summary[b])
                print_table(html_f, entangle_p, p_len[b], p_len[b], p[b, :], p[b, :], None)

                print("Intra")
                for i1 in range(p_len[b]):
                    for i2 in range(p_len[b]):
                        print("{0:.0f}\t".format(100*entangle_p[i1, i2]), end="")
                    print("")
                grouping = minimum_entangle(entangle_p)
                print(grouping.loss_group())
                group = grouping.group_by_count(int(p_len[b] * 0.6))
                for begin, end in group:
                    word_group = []
                    for i in range(begin,end):
                        word_group.append(word(p[b,i]))
                    print("[{}] ".format(" ".join(word_group)), end="")
                print("")

                print("Inter")

                for i_p in range(p_len[b]):
                    print("{}){}".format(i_p, word(p[b,i_p])), end=" ")
                print("")
                print("\t", end="")
                for i_h in range(h_len[b]):
                    print("{}){}".format(i_h, word(h[b,i_h])), end=" ")
                print("")

                for i_p in range(p_len[b]):
                    print("{}:\t".format(i_p), end="")
                    for i_h in range(h_len[b]):
                        print("{0:.2f}\t".format(100*entangle[i_p, i_h]), end="")
                    print("")
                print("Marginal ")
                entangle_m_p = np.sum(entangle, axis=1)
                entangle_m_h = np.sum(entangle, axis=0)
                print("< premise >")
                for i_p in range(p_len[b]):
                    print("{0:.0f}\t{1}".format(entangle_m_p[i_p], word(p[b,i_p])))
                print("< hypothesis >")
                for i_h in range(h_len[b]):
                    print("{0:.0f}\t{1}".format(entangle_m_h[i_h], word(h[b,i_h])))

            e_l, n_l, c_l = zip(*logit_array)
            print("e-n correlation : {}".format(pearsonr(e_l, n_l)))
            print("e-c correlation : {}".format(pearsonr(e_l, c_l)))
            print("n-c correlation : {}".format(pearsonr(n_l, c_l)))

            html_f.write("</body></html>\n")

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
            p, p_len, h, h_len, y = batch
            logits, = self.sess.run([soft_out], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
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
        print_shape("feature",feature)

        _, dim4 = feature.get_shape().as_list()
        dim = int(dim4 / 4)
        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h, self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            stop = [feature]
            E_all = []
            for label in range(3):
                E_all.append(de.explain('grad*input', soft_out[:, label], stop, x_input, xi))

            major = [[],[],[]]
            portion = [[],[],[],[],[]]
            for i in range(batch_size):
                print("-------")
                pred = np.argmax(run_logits, axis=1)

                true_label = D[y[i]]
                pred_label = D[pred[i]]

                print("--- {}({}) -- {} --- ".format(pred_label, true_label, run_logits[i]))
                for label in range(3):
                    r = E_all[label][0]
                    r_concat = np.sum(abs(r[i,0:dim*2]))
                    r_sub = np.sum(abs(r[i, dim*2:dim*3]))
                    r_odot = np.sum(abs(r[i, dim*3:dim*4]))
                    r_p = np.sum(abs(r[i,0:dim]))
                    r_h = np.sum(abs(r[i,dim:dim*2]))

                    if r_concat > r_sub and r_concat > r_odot:
                        major[label].append("concat")
                    elif r_sub > r_concat and r_sub > r_odot:
                        major[label].append("sub")
                    elif r_odot > r_concat and r_odot > r_sub:
                        major[label].append("odot")
                    else:
                        raise Exception("Unexpected")
                    r_sum = r_concat + r_sub + r_odot
                    portion[label].append((r_concat/r_sum, r_sub/r_sum, r_odot/r_sum, r_p/r_sum, r_h/r_sum))

                    print(D[label])
                    print("concat {0:.2f} ".format(r_concat))
                    for j in range(0,200):
                        print("{0:.2f}".format(r[i,j]*100), end=" ")
                    print()
                    print("sub {0:.2f} ".format(r_sub))
                    for j in range(dim*2,dim*2+200):
                        print("{0:.2f}".format(r[i,j]*100), end=" ")
                    print()
                    print("odot {0:.2f} ".format(r_odot))
                    for j in range(dim * 3, dim * 3 + 200):
                        print("{0:.2f}".format(r[i, j] * 100), end=" ")
                    print()

                    #for j in range(dim):
                    #    print("{0:.2f}".format(r[i,j]), end=" ")
                    #print("")

            for label in range(3):
                print(D[label])
                for operation in ["concat", "sub", "odot"]:
                    print("{} : {}".format(operation, major[label].count(operation)/batch_size))
                for portion_list in zip(*portion[label]):
                    print("{}".format(avg(portion_list)))




    def manaual_test(self, word2idx, idx2word):
        h1 = "It 's an interesting account of the violent history of modern Israel , and ends in the  Room where nine Jews were executed . "

    def interactive(self, word2idx):
        OOV = 0
        PADDING = 1
        max_sequence = 400

        def tokenize(sentence):
            def clean_str(string):
                """
                Tokenization/string cleaning for all datasets except for SST.
                """
                string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
                string = re.sub(r"\'s", " \'s", string)
                string = re.sub(r"\'ve", " \'ve", string)
                string = re.sub(r"n\'t", " n\'t", string)
                string = re.sub(r"\'re", " \'re", string)
                string = re.sub(r"\'d", " \'d", string)
                string = re.sub(r"\'ll", " \'ll", string)
                string = re.sub(r",", " , ", string)
                string = re.sub(r"!", " ! ", string)
                string = re.sub(r"\(", " \( ", string)
                string = re.sub(r"\)", " \) ", string)
                string = re.sub(r"\?", " \? ", string)
                string = re.sub(r" \'(.*)\'([ \.])", r" \1\2", string)
                string = re.sub(r"\s{2,}", " ", string)
                return string.strip().lower()

            tokens = clean_str(sentence).split(" ")
            return tokens

        def convert(tokens):
            OOV = 0
            l = []
            for t in tokens:
                if t in word2idx:
                    l.append(word2idx[t])
                else:
                    l.append(OOV)
                if len(l) == max_sequence:
                    break
            while len(l) < max_sequence:
                l.append(1)
            return np.array(l), len(tokens)


        def predict(sents):
            def transform(s):
                return convert(tokenize(s))
            data = []
            for sent1, sent2 in sents:
                p, p_len = transform(sent1)
                h, h_len = transform(sent2)
                data.append({
                    'p': p,
                    'p_len': p_len,
                    'h': h,
                    'h_len': h_len,
                    'y': 0})

            batches = get_batches(data, 100, 100)
            p, p_len, h, h_len, y = batches[0]
            logits, = self.sess.run([self.logits], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
            return logits
        terminate = False
        sents = []
        while not terminate:
            msg = input("Enter:")
            if msg == "!EOI":
                r = predict([(sents[0], sents[1]), (sents[1], sents[0])])
                print(r)
                sents = []
            elif msg == "!EXIT":
                terminate = True
            else:
                sents.append(msg)

    def run_server(self, word2idx):
        OOV = 0
        PADDING = 1
        max_sequence = 400
        from xmlrpc.server import SimpleXMLRPCServer
        from xmlrpc.server import SimpleXMLRPCRequestHandler

        def tokenize(sentence):
            def clean_str(string):
                """
                Tokenization/string cleaning for all datasets except for SST.
                """
                string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
                string = re.sub(r"\'s", " \'s", string)
                string = re.sub(r"\'ve", " \'ve", string)
                string = re.sub(r"n\'t", " n\'t", string)
                string = re.sub(r"\'re", " \'re", string)
                string = re.sub(r"\'d", " \'d", string)
                string = re.sub(r"\'ll", " \'ll", string)
                string = re.sub(r",", " , ", string)
                string = re.sub(r"!", " ! ", string)
                string = re.sub(r"\?", " ? ", string)
                string = re.sub(r" \'(.*)\'([ \.])", r" \1\2", string)
                string = re.sub(r"\s{2,}", " ", string)
                return string.strip().lower()

            tokens = clean_str(sentence).split(" ")
            return tokens

        def convert(tokens):
            OOV = 0
            l = []
            oov_cnt =0
            for t in tokens:
                if t in word2idx:
                    l.append(word2idx[t])
                else:
                    l.append(OOV)
                    oov_cnt += 1
                if len(l) == max_sequence:
                    break
            if oov_cnt > len(l) * 0.3 :
                print("WARNING : {} {}".format(oov_cnt, len(l)))
                print(tokens)
            while len(l) < max_sequence:
                l.append(1)

            return np.array(l), len(tokens)


        def predict(sents):
            def transform(s):
                return convert(tokenize(s))
            data = []
            for sent1, sent2 in sents:
                p, p_len = transform(sent1)
                h, h_len = transform(sent2)

                data.append({
                    'p': p,
                    'p_len': p_len,
                    'h': h,
                    'h_len': h_len,
                    'y': 0})
            batch_size = 100
            ori_len = len(data)
            if ori_len % batch_size:
                for j in range(batch_size - ori_len % batch_size):
                    data.append(data[-1])
            batches = get_batches(data, batch_size, 100)
            result = []
            idx = 0
            for batch in batches:
                p, p_len, h, h_len, y = batch
                if idx == 13:
                    for j in range(100):
                        loc = idx * 100 +j
                        if loc >= len(sents):
                            break

                idx += 1
                logits, = self.sess.run([self.logits], feed_dict={
                    self.input_p: p,
                    self.input_p_len: p_len,
                    self.input_h: h,
                    self.input_h_len: h_len,
                    self.input_y: y,
                    self.dropout_keep_prob: 1.0,
                })
                print("out")
                result += logits.tolist()
            print("return")
            return result[:ori_len]

        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)
        print("Preparing server")
        server = SimpleXMLRPCServer(("ingham.cs.umass.edu", 8125), requestHandler=RequestHandler)
        server.register_introspection_functions()

        server.register_function(predict, 'predict')
        print("Waiting")
        server.serve_forever()



    def run_adverserial(self, word2idx):
        test_cases, tag = adverserial.antonym()
        OOV = 0
        PADDING = 1
        max_sequence = 400
        def convert(tokens):
            OOV = 0
            l = []
            for t in tokens:
                if t in word2idx:
                    l.append(word2idx[t])
                else:
                    l.append(OOV)
                if len(l) == max_sequence:
                    break
            while len(l) < max_sequence:
                l.append(1)
            return np.array(l), len(tokens)

        data = []
        for test_case in test_cases:
            p, h, y = test_case
            p, p_len = convert(p)
            h, h_len = convert(h)
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
            p, p_len, h, h_len, y = batch
            logits, acc = self.sess.run([self.logits, self.acc], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
            print("Acc : {}".format(acc))
            for i, logit in enumerate(logits):
                #print(" ".join(test_cases[i][0]))
                print("----------")
                print(i)
                print(" ".join(test_cases[i][1]))
                print("y = {}({})".format(np.argmax(logit),test_cases[i][2]))
                print(logit)
                cate_total[tag[i]] += 1
                if np.argmax(logit) == test_cases[i][2] :
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
                    yp = [1,0,0]
                elif yi == 1:
                    yp = [0,1,0]
                else:
                    yp = [0,0,1]

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
            p, p_len, h, h_len, y = batch
            logits, = self.sess.run([self.logits], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
            return logits

        D = {0: "E",
             1: "N",
             2: "C"}

        print("view lrp")
        dev_batches = get_batches(dev_data, 100, 100)
        p, p_len, h, h_len, y = dev_batches[2]
        run_logits = run_result(dev_batches[2])
        print_shape("p", p)
        print_shape("p_len", p_len)
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="hypothesis:0")

        def print_color_html(word, r0, r1, r2, r_max, r_min):
            def normalize(val):
                v = (val - r_min) / (r_max - r_min) * 255
                assert( v < 256 and v >= 0 )
                return v
            normal_val = [normalize(r) for r in [r0, r1, r2]]
            bg_color = "".join(["%02x" % v for v in normal_val])
            if sum(normal_val) > 256 * 3 * 0.7:
                text_color = "000000" # black
            else:
                text_color = "ffffff" # white
            html = "<span style=\"color:#{}; background-color:#{}\">{}</span>&nbsp;\n".format(text_color, bg_color, word)
            return html

        with DeepExplain(session=self.sess) as de:

            x_input = [self.input_p, self.input_p_len, self.input_h, self.input_h_len, self.dropout_keep_prob]
            xi = [p, p_len, h, h_len, 1.0]
            yi = expand_y(y)
            stop = [p_emb_tensor, h_emb_tensor]
            soft_out = tf.nn.softmax(self.logits)

            c_e = soft_out[:, 2] - soft_out[:, 0]
            e_n = soft_out[:, 0] - soft_out[:, 1]
            C_E = de.explain('grad*input', c_e , stop, x_input, xi)
            E_N = de.explain('grad*input', e_n, stop, x_input, xi)

            E_all = []
            for label in range(3):
                E_all.append(de.explain('grad*input', soft_out[:, label], stop, x_input, xi))

            print("result----------")
            pred = np.argmax(run_logits, axis=1)
            f = open("result.html", "w")
            for i in range(100):
                print("-------")
                true_label = D[y[i]]
                pred_label = D[pred[i]]
                #if pred[i] == 2:
                #    E = C_E
                #else:
                #    E = E_N
                E_sum = list([[np.sum(E_all[label][s][i,:,:], axis=1) for s in range(2)] for label in range(3)])
                r_max = max([np.max(E_sum[label][s]) for label in range(3) for s in range(2)])
                r_min = min([np.min(E_sum[label][s]) for label in range(3) for s in range(2)])

                p_r = E_all[2][0] - E_all[0][0]
                h_r = E_all[2][1] - E_all[0][1]
                print("--- {}({}) -- {} --- ".format(pred_label, true_label, run_logits[i]))
                #print("sum[r]={} max={} min={}".format(np.sum(p_r[i])+ np.sum(h_r[i]), r_max, r_min))
                _, max_seq, _ = p_r.shape
                p_r_s = np.sum(p_r[i], axis=1)
                h_r_s = np.sum(h_r[i], axis=1)
                p_c_e_s = np.sum(C_E[0][i], axis=1)
                h_c_e_s = np.sum(C_E[1][i], axis=1)


                f.write("<html>")
                f.write("<div><span>Prediction={} , Truth={}</span><br>\n".format(pred_label, true_label))
                f.write("<p>Premise</p>\n")
                print("")
                print("premise: ")
                r_max = max([np.max(E_sum[label][s]) for label in range(3) for s in range(0,1)])
                r_min = min([np.min(E_sum[label][s]) for label in range(3) for s in range(0,1)])
                ce_max = max(np.max(p_c_e_s), -np.min(p_c_e_s))
                ce_min = -ce_max
                for j in range(max_seq):
                    print("{0}({1:.2f},{2:.2f})".format(word(p[i,j]), p_r_s[j], p_c_e_s[j]), end=" ")
                    #f.write(print_color_html(word(p[i,j]), E_sum[0][0][j], E_sum[1][0][j], E_sum[2][0][j], r_max, r_min))
                    v = p_c_e_s[j]
                    if v > 0:
                        r = v
                        b = 0
                    else:
                        r = 0
                        b = -v
                    f.write(print_color_html(word(p[i,j]), r, b, 0, ce_max, ce_min))

                print()
                _, max_seq, _ = h_r.shape
                f.write("<br><p>Hypothesis</p>\n")
                print("hypothesis: ")
                r_max = max([np.max(E_sum[label][s]) for label in range(3) for s in range(1, 2)])
                r_min = min([np.min(E_sum[label][s]) for label in range(3) for s in range(1, 2)])
                ce_max = max(np.max(h_c_e_s), -np.min(h_c_e_s))
                ce_min = -ce_max
                for j in range(max_seq):
                    print("{0}({1:.2f},{2:.2f})".format(word(h[i,j]), h_r_s[j], h_c_e_s[j]), end=" ")
                    #f.write(print_color_html(word(h[i,j]), E_sum[0][1][j], E_sum[1][1][j], E_sum[2][1][j], r_max, r_min))
                    v = h_c_e_s[j]
                    if v > 0:
                        r = v
                        b = 0
                    else:
                        r = 0
                        b = -v
                    f.write(print_color_html(word(h[i,j]), r, b, 0, ce_max, ce_min))
                print()
                f.write("</div><hr>")
            f.write("</html>")


    def print_time(self):
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

    def check_dev_plain(self, dev_data):
        dev_batches = get_batches(dev_data, 30, 100)
        acc_sum = []

        for batch in dev_batches:
            p, p_len, h, h_len, y = batch
            acc, loss, summary = self.sess.run([self.acc, self.loss, self.merged], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            })
            acc_sum.append(acc)
        print("Dev acc={} ".format(avg(acc_sum)))


    def check_dev(self, batches, g_step):
        acc_sum = []
        loss_sum = []
        step = 0
        for batch in batches:
            p, p_len, h, h_len, y = batch
            acc, loss, summary = self.sess.run([self.acc, self.loss, self.merged], feed_dict={
                self.input_p: p,
                self.input_p_len: p_len,
                self.input_h: h,
                self.input_h_len: h_len,
                self.input_y: y,
                self.dropout_keep_prob: 1.0,
            }, run_metadata=self.run_metadata)
            acc_sum.append(acc)
            loss_sum.append(loss)
            self.test_writer.add_summary(summary, g_step+step)
            step += 1
        acc = avg(acc_sum)
        if acc > self.best_acc:
            self.best_acc = acc
        print("Dev acc={} loss={} ".format(acc, avg(loss_sum)))

    def train(self, epochs, data, valid_data, rerun=False):
        print("Train")
        self.log_info()
        self.best_acc = 0
        if not rerun:
            self.sess.run(tf.global_variables_initializer())
        shuffle(data)
        batches = get_batches(data, self.batch_size, self.sent_crop_len)
        dev_batch_size = 200
        dev_batches = get_batches(valid_data, dev_batch_size, self.sent_crop_len)
        step_per_batch = int(len(data) / self.batch_size)
        log_every = int(step_per_batch/10)
        check_dev_every = int(step_per_batch/5)
        g_step = 0

        for i in range(epochs):
            print("Epoch {}".format(i))
            s_loss = 0
            l_acc = []
            time_estimator = TimeEstimator(len(batches), name="epoch")
            shuffle(batches)
            for batch in batches:
                g_step += 1
                p, p_len, h, h_len, y = batch
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                _, acc, loss, l2_loss, summary = self.sess.run([self.train_op, self.acc, self.loss, self.l2_loss, self.merged], feed_dict={
                    self.input_p: p,
                    self.input_p_len: p_len,
                    self.input_h: h,
                    self.input_h_len: h_len,
                    self.input_y: y,
                    self.dropout_keep_prob: 0.8,
                }, run_metadata=self.run_metadata, options=run_options)

                if g_step % log_every == 0 :
                    print("step{} : Loss={} L2_loss={} acc : {} ".format(g_step, loss, l2_loss, acc))
                if g_step % check_dev_every == 0 :
                    self.check_dev(dev_batches, g_step)
                s_loss += loss
                l_acc.append(acc)
                self.train_writer.add_summary(summary, g_step)
                self.train_writer.add_run_metadata(self.run_metadata, "meta_{}".format(g_step))
                time_estimator.tick()
                if math.isnan(loss):
                    raise Exception("Nan loss reported")
            current_step = tf.train.global_step(self.sess, self.global_step)
            path = self.saver.save(self.sess, self.save_path(), global_step=current_step)
            print("Checkpoint saved at {}".format(path))
            print("Training Average loss : {} , acc : {}".format(s_loss, avg(l_acc)))
        print("Best dev acc={}".format(self.best_acc))