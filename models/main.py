import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("LD_LIBRARY_PATH : {}".format(os.environ["LD_LIBRARY_PATH"]))
os.environ["CUDA_HOME"] = "/usr/local/cuda"

from collections import Counter

from models.FAIRmodel import FAIRModel
from models.data_manager import *
from models.manager import *
from models.analysis import *
tf.logging.set_verbosity(tf.logging.INFO)


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()


def build_voca(path):
    # word is valid if count > 10 or exists in GLOVE
    run_size = 100
    voca = Counter()
    glove_voca_list = glove_voca()
    max_length = 0
    mnli_train = load_nli_data(path)
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])
        for token in s1_tokenize + s2_tokenize:
            voca[token] += 1

        if len(s1_tokenize) > max_length:
            max_length = len(s1_tokenize)
        if len(s2_tokenize) > max_length:
            max_length = len(s2_tokenize)
    print(len(voca))
    print("Max length : {}".format(max_length))

    word2idx = dict()
    word2idx["<OOV>"] = 0
    word2idx["<PADDING>"] = 1
    idx = 2
    glove_found = 0
    for word, count in voca.items():
        if count > 10 or word in glove_voca_list:
            word2idx[word] = idx
            idx += 1
        if word in glove_voca_list:
            glove_found += 1
    print(len(word2idx))
    print("Glove found : {}".format(glove_found))
    return word2idx


def voca_stat(path):
    mnli_train = load_nli_data(path)
    voca = Counter()
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])
        for token in s1_tokenize + s2_tokenize:
            voca[token] += 1
    save_pickle("word_count", voca)
    fout = open("word_count.txt", "w")
    for word, count in voca.most_common():
        fout.write("{}\t{}\n".format(word, count))
    fout.close()



def load_voca():
    if args.corpus_name == "multinli":
        return load_pickle("word2idx")
    else:
        return load_pickle("word2idx_{}".format(args.corpus_name))


def transform_corpus(path, save_path, max_sequence = 400):
    voca = load_voca()
    mnli_train = load_nli_data(path)
    def convert(tokens):
        OOV = 0
        l = []
        for t in tokens:
            if t in voca:
                l.append(voca[t])
            else:
                l.append(OOV)
            if len(l) == max_sequence:
                break
        while len(l) < max_sequence:
            l.append(1)
        return np.array(l), len(tokens)

    data = []
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])

        s1, s1_len = convert(s1_tokenize)
        s2, s2_len = convert(s2_tokenize)
        label = datum["label"]
        y = label
        data.append({
            'p': s1,
            'p_len': s1_len,
            'h': s2,
            'h_len': s2_len,
            'y': y})

    for _ in range(10000):
        idx1 = random.randint(0, len(mnli_train))
        idx2 = random.randint(0, len(mnli_train))
        datum1 = mnli_train[idx1]
        datum2 = mnli_train[idx2]
        s1_tokenize = tokenize(datum1['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum2['sentence2_binary_parse'])

        s1, s1_len = convert(s1_tokenize)
        s2, s2_len = convert(s2_tokenize)
        y = 1

        data.append({
            'p': s1,
            'p_len': s1_len,
            'h': s2,
            'h_len': s2_len,
            'y': y})

    save_pickle(save_path, data)
    return data


def train_fair():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca, batch_size=args.batch_size, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    data = load_pickle("train_corpus.pickle")
    validate = load_pickle("dev_corpus")
    epochs = 10
    model.train(epochs, data, validate)

def load_dev_data():
    if args.corpus_name=="snli":
        return load_pickle("dev_snli")
    elif args.corpus_name=="multinli":
        return load_pickle("multinli_dev")


def load_mnli_explain(max_sequence = 400):
    return load_pickle("mnli_explain")
    voca = load_voca()
    mnli_data = load_nli_data(path_dict["dev_matched"])
    explation = load_nli_explain(path_dict["mnli_explain"])

    def find(prem, hypothesis):
        for datum in mnli_data:
            if prem == datum['sentence1'].strip() and hypothesis == datum['sentence2'].strip():
                return datum
        print("Not found")
        raise Exception(prem)

    def convert(tokens):
        OOV = 0
        l = []
        for t in tokens:
            if t in voca:
                l.append(voca[t])
            else:
                l.append(OOV)
            if len(l) == max_sequence:
                break
        while len(l) < max_sequence:
            l.append(1)
        return np.array(l), len(tokens)

    def token_match(tokens1, tokens2):
        gold_indice = []
        for token in tokens1:
            matches = []
            for idx, t in enumerate(tokens2):
                if token == t:
                    matches.append(idx)
            if len(matches) == 1:
                gold_indice.append(matches[0])
            else:
                for idx, t in enumerate(tokens2):
                    print((idx, t), end =" ")
                print(token)
                print(matches)
                print("Select indice: " , end="")
                user_written = input()
                gold_indice += [int(t) for t in user_written.split()]
        return gold_indice

    data = []
    for entry in explation:
        p, h, pe, he = entry

        datum = find(p.strip(),h.strip())

        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])

        e_indice_p = token_match(pe, s1_tokenize)
        e_indice_h = token_match(he, s2_tokenize)

        s1, s1_len = convert(s1_tokenize)
        s2, s2_len = convert(s2_tokenize)
        label = datum["label"]
        y = label
        data.append({
            'p': s1,
            'p_len': s1_len,
            'h': s2,
            'h_len': s2_len,
            'y': y,
            'p_explain':e_indice_p,
            'h_explain':e_indice_h
        })

    save_pickle("mnli_explain", data)
    return data


def train_cafe():
    voca = load_voca()
    model = Manager(max_sequence=args.max_sequence, word_indice=voca, batch_size=args.batch_size,
                    num_classes=3, vocab_size=1000,
                    embedding_size=300, lstm_dim=1024)
    if args.corpus_name=="snli":
        train = load_pickle("snli_train")
        validate = load_pickle("dev_snli")
    elif args.corpus_name=="multinli":
        train = load_pickle("multinli_train")
        validate = load_pickle("multinli_dev")
    sanity_check(train)

    epochs = 30
    model.train(epochs, train, validate)

def sanity_check(data):
    for entry in data:
        v = entry['y']
        assert(v==0 or v==1 or v==2)
    print("Sane")


def train_keep_cafe():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    data = load_pickle("train_corpus.pickle")
    validate = load_dev_data()
    manager.load("model-15340")
    manager.train(20, data, validate, True)



def lrp_run():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
#    manager.load("hdrop2/model-41418")
    manager.load("hdrop/model-42952")

    validate = load_dev_data()
    #manager.view_lrp(validate, reverse_index(voca))
    #manager.lrp_3way(validate, reverse_index(voca))
    #manager.view_weights(validate)

    #manager.feature_invertor(validate, reverse_index(voca))
    manager.GI_input_layer(validate, reverse_index(voca))
    #manager.lrp_entangle(validate, reverse_index(voca))

    #manager.lrp_premise(validate, reverse_index(voca))

def view_weights():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
#    manager.load("wattention/model-12272")
    manager.load("hdrop/model-42952")
    #manager.load("hdrop2/model-41418")
    voca = load_voca()

    validate = load_dev_data()
    manager.check_dev_plain(validate)
    manager.corr_analysis(validate, reverse_index(voca))



def sa_run():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca, batch_size=args.batch_size, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    model.load("model-13091")
    validate = load_pickle("dev_corpus")
    model.sa_analysis(validate[:100], reverse_index(voca))


def view_weight_fair():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca, batch_size=args.batch_size, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    model.load("model-13091")
    validate = load_pickle("dev_corpus")
    model.view_weights(validate)


def run_adverserial():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    #manager.load("hdrop2/model-41418")
    manager.load("hdrop/model-42952")
    manager.run_adverserial(voca)

def init_manager():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    return manager

def eval_lrp():
    manager = init_manager()
    manager.load("hdrop/model-42952")
    manager.eval_lrp_feature(load_dev_data())

def poly_analyze():
    manager = init_manager()
    manager.load("248/model-55224")
    validate = load_dev_data()

    manager.poly_weights(validate)


def analyze():
    A =Analyzer()
    #A.multi_analysis()
    A.boy_girl()



def interactive():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    #manager.load("hdrop2/model-41418")
    manager.load("hdrop/model-42952")
    manager.interactive(voca)

def run_server():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    # manager.load("hdrop2/model-41418")
    #manager.load("hdrop/model-42952")
    manager.load("240/model-15735")
    manager.run_server(voca)


def eval_explain():
    data = load_mnli_explain(400)
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    # manager.load("hdrop2/model-41418")
    #manager.load("hdrop/model-42952")
    manager.load("240/model-15735")
    manager.explain_eval(data, reverse_index(voca))

if __name__ == "__main__":
    action = "eval_explain"
    if "build_voca" in action:
        word2idx = build_voca(path_dict["training_snli"])
        corpus_name = args.corpus_name
        save_pickle("word2idx_{}".format(corpus_name), word2idx)

    if "voca_stat" in action:
        voca_stat(path_dict["training_mnli"])

    # reformat corpus
    if "transform" in action:
        transform_corpus(path_dict["training_mnli"], "multinli_train_pp")

    if "train_fair" in action:
        train_fair()

    if "train_cafe" in action:
        train_cafe()

    if "sa_run" in action:
        sa_run()

    if "poly_analyze" in action:
        poly_analyze()

    if "lrp_run" in action:
        lrp_run()

    if "train_keep_cafe" in action:
        train_keep_cafe()

    if "run_adverserial" in action:
        run_adverserial()

    if "eval_lrp" in action:
        eval_lrp()

    if "analyze" in action:
        analyze()

    if "interactive" in action:
        interactive()

    if "run_server" in action:
        run_server()

    if "eval_explain" in action:
        eval_explain()