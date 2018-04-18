
from collections import Counter

from models.FAIRmodel import FAIRModel
from models.common import *
from models.data_manager import *
from models.manager import *

tf.logging.set_verbosity(tf.logging.INFO)

def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

def build_voca():
    # word is valid if count > 10 or exists in GLOVE
    run_size = 100
    voca = Counter()
    glove_voca_list = glove_voca()
    max_length = 0
    mnli_train = load_nli_data(path_dict["training_mnli"])
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

def load_voca():
    return load_pickle("word2idx")


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
            'p':s1,
            'p_len':s1_len,
            'h':s2,
            'h_len':s2_len,
            'y':y})

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


def train_cafe():
    voca = load_voca()
    model = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                    num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    data = load_pickle("train_corpus.pickle")
    validate = load_pickle("dev_corpus")
    epochs = 10
    model.train(epochs, data, validate)



def sa_run():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca, batch_size=args.batch_size, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    model.load("model-13091")
    validate = load_pickle("dev_corpus")
    model.sa_analysis(validate[:100], reverse_index(voca))

def view_weight():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca, batch_size=args.batch_size, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    model.load("model-13091")
    validate = load_pickle("dev_corpus")
    model.view_weights(validate)


if __name__ == "__main__":
    action = "train_cafe"
    if "build_voca" in action:
        word2idx = build_voca()
        save_pickle("word2idx", word2idx)

    # reformat corpus
    if "transform" in action:
        transform_corpus(path_dict["dev_matched"], "dev_corpus")

    if "train_fair" in action:
        train_fair()

    if "train_cafe" in action:
        train_cafe()

    if "sa_run" in action:
        sa_run()

    if "view_weight" in action:
        view_weight()