import os
from parameter import args
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

from collections import Counter

from models.FAIRmodel import FAIRModel
from models.data_manager import *
from models.manager import *
from models.common import construct_one_hot_feature_tensor
tf.logging.set_verbosity(tf.logging.INFO)

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}


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

def generate_pos_feature_tensor(parses, left_padding_and_cropping_pairs):
    pos_vectors = []
    for parse in parses:
        pos = parsing_parse(parse)
        pos_vector = [(idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
        pos_vectors.append(pos_vector)

    return construct_one_hot_feature_tensor(pos_vectors, left_padding_and_cropping_pairs, 2, column_size=len(POS_Tagging))

def parsing_parse(parse):
    base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
    pos = [pair.split(" ")[0] for pair in base_parse]
    return pos


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
    shared_content = load_mnli_shared_content() 
    premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] #* args.batch_size
    for datum in mnli_train:
        pair_id = datum['pairID']
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])

#        p_pos = generate_pos_feature_tensor(datum['sentence1_parse'], premise_pad_crop_pair)
#        h_pos = generate_pos_feature_tensor(datum['sentence2_parse'], hypothesis_pad_crop_pair)

        p_exact = construct_one_hot_feature_tensor([shared_content[pair_id]["sentence1_token_exact_match_with_s2"][:]], premise_pad_crop_pair, 1)
        h_exact = construct_one_hot_feature_tensor([shared_content[pair_id]["sentence2_token_exact_match_with_s1"][:]], hypothesis_pad_crop_pair, 1)

        s1, s1_len = convert(s1_tokenize)
        s2, s2_len = convert(s2_tokenize)
        label = datum["label"]
        y = label
        data.append({
            'p': s1,
            'p_pos': datum['sentence1_parse'],
            'p_exact': p_exact,
            'h': s2,
            'h_pos': datum['sentence2_parse'],
            'h_exact': h_exact,
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


def train_cafe():
    voca = load_voca()
    model = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                    num_classes=3, vocab_size=1000,
                    embedding_size=300, lstm_dim=1024)
    data = load_pickle("train_corpus.pickle")
    validate = load_pickle("dev_corpus")
    epochs = 30
    model.train(epochs, data, validate)


def train_keep_cafe():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    data = load_pickle("train_corpus.pickle")
    validate = load_pickle("dev_corpus")

    manager.load("model-15340")
    manager.train(20, data, validate, True)



def lrp_run():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    manager.load("hdrop2/model-41418")
    validate = load_pickle("dev_corpus")

    #manager.lrp_3way(validate, reverse_index(voca))
    manager.lrp_entangle(validate, reverse_index(voca))

def view_weights():
    voca = load_voca()
    manager = Manager(max_sequence=100, word_indice=voca, batch_size=args.batch_size,
                      num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    # Dev acc=0.6576999819278717 loss=0.8433943867683411
    manager.load("wattention/model-12272")
    validate = load_pickle("dev_corpus")

    manager.view_weights(validate)



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


if __name__ == "__main__":
    actions = ["transform", "train_cafe"]
    if "build_voca" in actions:
        word2idx = build_voca()
        save_pickle("word2idx", word2idx)

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
