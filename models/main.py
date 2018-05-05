from collections import Counter
import json
import numpy as np
import os
import re

from parameter import args
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)
import tensorflow as tf

from models.FAIRmodel import FAIRModel
import models.data_manager
from models.manager import Manager
import models.common
from parameter import args, path_dict

tf.logging.set_verbosity(tf.logging.INFO)


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()


def build_voca():
    # word is valid if count > 10 or exists in GLOVE
    voca = Counter()
    char_counter = Counter()
    glove_voca_list = models.common.glove_voca()
    max_length = 0
    mnli_train = models.data_manager.load_nli_data(path_dict["training_mnli"])
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])
        for token in s1_tokenize + s2_tokenize:
            voca[token] += 1
            for c in token:
                char_counter[c] += 1

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

    char_indices = dict(zip(char_counter, range(len(char_counter))))
    return word2idx, char_indices


def load_voca():
    return models.common.load_pickle("word2idx")


def load_char_length():
    return len(models.common.load_pickle("charidx"))


def load_charidx():
    return models.common.load_pickle("charidx")


def load_shared_content(fh, shared_content):
    for line in fh:
        row = line.rstrip().split("\t")
        key = row[0]
        value = json.loads(row[1])
        shared_content[key] = value


def load_mnli_shared_content():
    shared_file_exist = False
    # shared_path = config.datapath + "/shared_2D_EM.json"
    # shared_path = config.datapath + "/shared_anto.json"
    # shared_path = config.datapath + "/shared_NER.json"
    shared_path = path_dict["shared_mnli"]
    # shared_path = "../shared.json"
    print(shared_path)
    if os.path.isfile(shared_path):
        shared_file_exist = True
    # shared_content = {}
    assert shared_file_exist
    # if not shared_file_exist and config.use_exact_match_feature:
    #     with open(shared_path, 'w') as f:
    #         json.dump(dict(reconvert_shared_content), f)
    # elif config.use_exact_match_feature:
    with open(shared_path) as f:
        shared_content = {}
        load_shared_content(f, shared_content)
        # shared_content = json.load(f)
    return shared_content


def transform_corpus(path, save_path):
    voca = load_voca()
    charidx = load_charidx()
    args.char_vocab_size = load_char_length()
    mnli_train = models.data_manager.load_nli_data(path)

    data = []
    shared_content = load_mnli_shared_content()
    premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0, 0)]
    for datum in mnli_train:
        pair_id = datum['pairID']
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])

        p_exact = models.common.construct_one_hot_feature_tensor([
            shared_content[pair_id]["sentence1_token_exact_match_with_s2"][:]],
                                                   premise_pad_crop_pair, 1)
        h_exact = models.common.construct_one_hot_feature_tensor([
            shared_content[pair_id]["sentence2_token_exact_match_with_s1"][:]],
                                                   hypothesis_pad_crop_pair, 1)

        s1, s1_len = models.common.convert_tokens(s1_tokenize, voca)
        s2, s2_len = models.common.convert_tokens(s2_tokenize, voca)
        label = datum["label"]
        y = label
        data.append({
            'p': s1,
            'p_pos': datum['sentence1_parse'],
            'p_exact': p_exact.T,
            'p_char': get_char_index(datum['sentence1_binary_parse'], charidx),
            'p_len': s1_len,
            'h': s2,
            'h_pos': datum['sentence2_parse'],
            'h_exact': h_exact.T,
            'h_char': get_char_index(datum['sentence2_binary_parse'], charidx),
            'h_len': s2_len,
            'y': y})

    models.common.save_pickle(save_path, data)
    return data


def get_char_index(tk, char_indices):
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    token_sequence = tokenize(tk)
    data = np.zeros((args.max_sequence, args.char_in_word_size),
                    dtype=np.int32)

    for i in range(args.max_sequence):
        if i >= len(token_sequence):
            continue
        else:
            chars = [c for c in token_sequence[i]]
            for j in range(args.char_in_word_size):
                if j >= (len(chars)):
                    break
                else:
                    index = char_indices[chars[j]]
                data[i, j] = index

    return data


def train_fair():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    data = models.common.load_pickle("train_corpus.pickle")
    validate = models.common.load_pickle("dev_corpus")
    epochs = 10
    model.train(epochs, data, validate)


def train_cafe():
    voca = load_voca()
    args.char_vocab_size = load_char_length()
    model = Manager(max_sequence=100, word_indice=voca,
                    batch_size=args.batch_size, num_classes=3, vocab_size=1000,
                    embedding_size=300, lstm_dim=1024)
    data = models.common.load_pickle("train_corpus.pickle")
    validate = models.common.load_pickle("dev_corpus")
    epochs = 30
    model.train(epochs, data, validate)


def train_keep_cafe():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    data = models.common.load_pickle("train_corpus.pickle")
    validate = models.common.load_pickle("dev_corpus")

    manager.load("model-15340")
    manager.train(20, data, validate, True)


def lrp_run():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    manager.load("hdrop2/model-41418")
    validate = models.common.load_pickle("dev_corpus")
    # manager.view_lrp(validate, reverse_index(voca))
    # manager.lrp_3way(validate, reverse_index(voca))
    manager.lrp_entangle(validate, models.common.reverse_index(voca))


def view_weights():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    manager.load("wattention/model-12272")
    validate = models.common.load_pickle("dev_corpus")

    manager.view_weights(validate)


def sa_run():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    model.load("model-13091")
    validate = models.common.load_pickle("dev_corpus")
    model.sa_analysis(validate[:100], models.common.reverse_index(voca))


def view_weight_fair():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    model.load("model-13091")
    validate = models.common.load_pickle("dev_corpus")
    model.view_weights(validate)


def run_adverserial():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca,
                      batch_size=args.batch_size, num_classes=3,
                      vocab_size=1000, embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    # manager.load("hdrop2/model-41418")
    manager.load("hdrop/model-42952")
    manager.run_adverserial(voca)


if __name__ == "__main__":
    actions = ["train_cafe"]
    if "build_voca" in actions:
        word2idx, charidx = build_voca()
        models.common.save_pickle("word2idx", word2idx)
        models.common.save_pickle("charidx", charidx)

    # reformat corpus
    if "transform" in actions:
        transform_corpus(path_dict["dev_matched"], "dev_corpus")
        transform_corpus(path_dict["training_mnli"], "train_corpus.pickle")

    if "train_fair" in actions:
        train_fair()

    if "train_cafe" in actions:
        train_cafe()

    if "sa_run" in actions:
        sa_run()

    if "view_weights" in actions:
        view_weights()

    if "lrp_run" in actions:
        lrp_run()

    if "train_keep_cafe" in actions:
        train_keep_cafe()

    if "run_adverserial" in actions:
        run_adverserial()
