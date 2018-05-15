import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from parameter import *

from models.FAIRmodel import FAIRModel
from models.data_manager import *
from models.manager import *


def load_voca():
    if args.corpus_name == "multinli":
        return load_pickle("word2idx")
    elif args.corpus_name == "snli":
        return load_pickle("word2idx_{}".format(args.corpus_name))
    else:
        raise Exception("No corpus_name")

def load_dev_data():
    if args.corpus_name=="snli":
        return load_pickle("dev_snli")
    elif args.corpus_name=="multinli":
        return load_pickle("multinli_dev")


class Analyzer():
    def __init__(self):
        voca = load_voca()
        self.manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                          num_classes=3, vocab_size=1000,
                          embedding_size=300, lstm_dim=1024)
        # Dev acc=0.6576999819278717 loss=0.8433943867683411
        #self.manager.load("hdrop2/model-41418")
        self.manager.load("hdrop/model-42952")
        #self.manager.load("211/model-36816")
        self.validate = load_dev_data()
        self.batch_size = 30
        max_seq = 100
        self.dev_batches = get_batches(self.validate, self.batch_size, max_seq)
        self.word2idx = voca

    def sess(self):
        return self.manager.sess


    def get_high2_out(self):
        return tf.get_default_graph().get_tensor_by_name(name="pred/high2/y:0")

    def get_multi_unit(self):
        return tf.get_default_graph().get_tensor_by_name(name="multiply_layer/multi_dense/xw_plus_b:0")

    def get_input_emb(self):
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="hypothesis:0")
        return [p_emb_tensor, h_emb_tensor]

    def common_input(self):
        x_input = [self.manager.input_p,
                   self.manager.input_p_len,
                   self.manager.input_h,
                   self.manager.input_h_len,
                   self.manager.dropout_keep_prob]
        return x_input

    def analyze_batch0_end(self, stop):
        PREMISE = 0
        HYPOTHESIS = 1

        p, p_len, h, h_len, y = self.dev_batches[0]
        xi = [p, p_len, h, h_len, 1.0]
        E_list =[]
        with DeepExplain(session=self.sess(), graph=self.sess().graph) as de:
            target = tf.nn.softmax(self.manager.logits)
            for label in range(3):
                raw_E = de.explain('grad*input', target[:,label], stop, self.common_input(), xi)
                E_list.append(raw_E)
        return E_list


    def analyze_batch0(self, stop, target):
        PREMISE = 0
        HYPOTHESIS = 1

        max_seq = 100
        dev_batches = get_batches(self.validate, self.batch_size, max_seq)
        p, p_len, h, h_len, y = dev_batches[0]
        x_input = [self.manager.input_p,
                   self.manager.input_p_len,
                   self.manager.input_h,
                   self.manager.input_h_len,
                   self.manager.dropout_keep_prob]
        xi = [p, p_len, h, h_len, 1.0]

        with DeepExplain(session=self.sess(), graph=self.sess().graph) as de:
            raw_E = de.explain('grad*input', target, stop, x_input, xi)
        return raw_E


    def multi_analysis(self):
        D = {0: "E",
             1: "N",
             2: "C"}

        #  analyze the mutli vector's contribution to classification
        m_vector = self.get_multi_unit()

        R = self.analyze_batch0_end([self.get_multi_unit(), self.get_high2_out()])
        print(len(R))
        print(len(R[0]))
        logit_out = self.manager.run_result(self.dev_batches[0])

        p, p_len, h, h_len, y = self.dev_batches[0]
        for i in range(self.batch_size):
            pred_label = D[np.argmax(logit_out[i])]
            true_label = D[y[i]]
            print("pred = {}({})".format(pred_label, true_label))
            for label in range(3):
                head = np.sum(R[label][0][i])
                tail = np.sum(R[label][1][i])
                print("{} ".format(D[label]) ,end="")
                print("{0:.2f} {1:.2f}".format(head, tail), end=" ")
            print("")
        # TODO analyze the input tokens contributing to multi vector
        #R = self.analyze_batch0(self.get_input_emb(), m_vector[:612 * 4])
        #for i in range(self.manager.batch_size):
        #    print()
    def text2idx(self, sent, max_sequence):
        tokens = sent.split(" ")
        OOV = 0
        l = []
        for t in tokens:
            if t in self.word2idx:
                l.append(self.word2idx[t])
            else:
                l.append(OOV)
            if len(l) == max_sequence:
                break
        while len(l) < max_sequence:
            l.append(1)
        return np.array(l), len(tokens)

    @staticmethod
    def batch2xi(batch):
        p, p_len, h, h_len, y = batch
        xi = [p, p_len, h, h_len, 1.0]
        return xi

    def boy_girl(self):
        D = {0: "E",
             1: "N",
             2: "C"}

        prem = "I am a boy ."
        hypo = "I am a girl ."
        #prem = "and the professors who go there and you're not going to see the professors you know you're going to see some TA you know uh"
        #hypo = "You don't really see the TAs."
        max_sequence = 100
        p, p_len = self.text2idx(prem, max_sequence)
        h, h_len = self.text2idx(hypo, max_sequence)
        y = 2
        data = []
        batch_size = 1
        for j in range(batch_size):
            data.append({
                'p': p,
                'p_len': p_len,
                'h': h,
                'h_len': h_len,
                'y': y})

        batches = get_batches(data, batch_size, max_sequence)
        xi = self.batch2xi(batches[0])
        logit_out = self.manager.run_result(batches[0])
        print("logit : {}".format(logit_out[0]))
        p_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="premise:0")
        h_emb_tensor = tf.get_default_graph().get_tensor_by_name(name="hypothesis:0")
        stop = [p_emb_tensor, h_emb_tensor]

        p, p_len, h, h_len, y = batches[0]

        stop_val = self.manager.sess.run(stop, feed_dict={
            self.manager.input_p: p,
            self.manager.input_p_len: p_len,
            self.manager.input_h: h,
            self.manager.input_h_len: h_len,
            self.manager.input_y: y,
            self.manager.dropout_keep_prob: 1.0,
        })

        with DeepExplain(session=self.sess()) as de:
            stop = self.get_input_emb()
            x_input = self.common_input()
            E_list = []
            for label in range(3):
                PREMISE = 0
                HYPOTHESIS = 1
                raw_E = de.explain('intgrad', self.manager.logits[:,label], stop, x_input, xi, stop_val =stop_val)
                raw_E[PREMISE] = np.sum(raw_E[PREMISE], axis=2)
                raw_E[HYPOTHESIS] = np.sum(raw_E[HYPOTHESIS], axis=2)
                E_list.append(raw_E)
        for b in range(batch_size):
            for label in range(3):
                print(D[label])
                for i in range(p_len):
                    print("{0:.2f}".format(E_list[label][PREMISE][b,i]), end="\t")
                print("")
                for i in range(h_len):
                    print("{0:.2f}".format(E_list[label][HYPOTHESIS][b,i]), end="\t")
                print("")

    def GI_word(self):
        chi_ce = load_pickle("chi_ce")
        f_dict = {}
        for f_id, entry in chi_ce:
            info = {
                'cut': entry[1],
                'high': entry[2]
            }
            f_dict[f_id] = info

        def fliped_GI(feature_before, feature_after):
            #             return x2, feature_avg, pp_actual>pp_expected
            dim = 612

            f_list = []
            for f in range(dim * 4):
                if f in f_dict:
                    info = f_dict[f]
                    def is_active(v):
                        return (v > info['cut'] and info['high']) or (v <= info['cut'] and not info['high'])
                    if is_active(feature_before[f]) and not is_active(feature_after[f]):
                        f_list.append(f)
            return f_list

        def count_GI(feature):
            #             return x2, feature_avg, pp_actual>pp_expected
            dim = 612
            cnt = 0
            for f in range(dim * 4):
                if f in f_dict:
                    info= f_dict[f]
                    active =  (feature[f] > info['cut'] and info['high']) or (feature[f] <= info['cut'] and not info['high'])
                    if active:
                        cnt += 1
            return cnt


        def get_interest_entry():
            interest_entry = []
            # TODO run to get featrue
            for batch in self.dev_batches:
                p, p_len, h, h_len, y = batch
                logit, feature = self.manager.run_w_feature(batch)
                batch_len = len(p_len)
                for b in range(batch_len):
                    if count_GI(feature[b]) > 0 :
                        entry = [
                            p[b],
                            p_len[b],
                            h[b],
                            h_len[b],
                            y[b],
                        ]
                        interest_entry.append(entry)
            return interest_entry

        def entry_to_perturb_batch(entry):
            OOV = 1
            p, p_len, h, h_len, y = entry

            removed_words = [-1]
            new_batch = []
            new_entry = {
                'p': p,
                'p_len': p_len,
                'h': h,
                'h_len': h_len,
                'y': y}
            new_batch.append(new_entry)


            for i in range(h_len):
                for j in range(p_len):
                    new_h = np.array(h, copy=True)
                    new_h[i] = OOV
                    new_p = np.array(p, copy=True)
                    new_p[j] = OOV

                    removed_words.append((p[j],h[i]))
                    new_entry = {
                            'p': p,
                            'p_len': p_len,
                            'h': new_h,
                            'h_len': h_len,
                            'y': y}
                    new_batch.append(new_entry)

            # TODO repeat for premise
            return new_batch, removed_words

        idx2word = reverse_index(self.word2idx)

        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"

        def idx2sent(sent):

            text= ""
            sent_len = len(sent)
            for i in range(sent_len):
                text += (word(sent[i]) + " ")
            return text

        #print("Getting interest entry")
        #interest_entry = get_interest_entry()
        #save_pickle("interest_entry", interest_entry)
        interest_entry = load_pickle("interest_entry")

        def demo1(entry):
            print(idx2sent(entry[0]))
            print(idx2sent(entry[2]))
            raw_batch, rem_words = entry_to_perturb_batch(entry)
            batches = get_batches(raw_batch, len(raw_batch), 100)
            for batch in batches:
                p, p_len, h, h_len, y = batch
                logit, feature = self.manager.run_w_feature(batch)
                for b in range(len(p_len)):
                    f_count = count_GI(feature[b,:])
                    hypo = idx2sent(h[b,:])
                    print("{} f_count={} / {}".format(logit[b], f_count, hypo))

        fdim = 612
        flip_dict = {}
        trial = []
        def count(entry):
            raw_batch, rem_words = entry_to_perturb_batch(entry)
            batches= get_batches(raw_batch, len(raw_batch), 100)

            def get_flip_entry(f):
                if f not in flip_dict:
                    flip_dict[f] = []
                return flip_dict[f]


            idx = 0
            for batch in batches:
                p, p_len, h, h_len, y = batch
                logit, feature = self.manager.run_w_feature(batch)
                idx += 1
                for b in range(1, len(p_len)):
                    f_list = fliped_GI(feature[0], feature[b])
                    for f in f_list:
                        d = get_flip_entry(f)
                        d.append(rem_words[idx])
                    idx += 1
            trial.append(rem_words)
            # return number of times each word acted as switch for each dimension in the

        for entry in interest_entry[:1000]:
            count(entry)

        trial = [item for sublist in trial for item in sublist]
        bg_count = Counter(trial)
        all_words = []
        for l in flip_dict.values():
            all_words = all_words + l

        activity = []
        for idx, occur in Counter(all_words).items():
            activity.append((idx, occur, bg_count[idx]))

        activity.sort(key=lambda x:x[1], reverse=True)
        save_pickle("activity", activity)
        boring = load_pickle("neg_high")
        for idx, flip, total in activity[:1000]:
            a,b = idx
            probability = flip / total
            if total >  100 and probability > 5 and a not in boring and b not in boring:
                print("word={} p={} count={}".format(word(idx), probability, total))

    def view_words(self):
        idx2word = reverse_index(self.word2idx)

        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"
        activity = load_pickle("activity")
        view = [(idx, flip/total) for idx, flip, total in activity if total > 100]
        view.sort(key=lambda x:x[1], reverse=True)
        for idx, p in view[:50]:
            print("{} {}".format(word(idx), p))
        save_pickle("neg_high", list([idx for idx, p in view[:50]]))

    def view_pairs(self):
        idx2word = reverse_index(self.word2idx)

        def word(index):
            if index in idx2word:
                if idx2word[index] == "<PADDING>":
                    return "PADDING"
                else:
                    return idx2word[index]
            else:
                return "OOV"
        activity = load_pickle("activity")
        boring = load_pickle("neg_high")
        view = [(idx, flip/total) for idx, flip, total in activity if total > 2]
        count = dict()
        for idx, flip, total in activity:
            count[idx] = total

        view.sort(key=lambda x:x[1], reverse=True)
        for idx, p in view:
            a,b = idx
            if a not in boring and b not in boring:
                print("word={},{} p={} count={}".format(word(a), word(b), p, count[idx]))

if __name__ == "__main__":
    A = Analyzer()
    A.view_pairs()
