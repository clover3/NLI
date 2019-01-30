import json
import random
import re
import csv

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

def label2sym(label):
    D = {0:"E",
    1:"N",
    2:"C"}
    return D[label]


def load_nli_data(path, snli=False, shuffle=True):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data.
    """
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    # data is list of dict
    return data

def load_nli_data_genre(path, genre, snli=True, shuffle = True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data


def worker(shared_content, dataset):
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    for example in dataset:
        s1_tokenize = tokenize(example['sentence1_binary_parse'])
        s2_tokenize = tokenize(example['sentence2_binary_parse'])

        s1_token_exact_match = [0] * len(s1_tokenize)
        s2_token_exact_match = [0] * len(s2_tokenize)
        s1_token_antonym = [0] * len(s1_tokenize)
        s2_token_antonym = [0] * len(s2_tokenize)
        for i, word in enumerate(s1_tokenize):
            matched = False
            for j, w2 in enumerate(s2_tokenize):
                matched = is_exact_match(word, w2)
                if matched:
                    s1_token_exact_match[i] = 1
                    s2_token_exact_match[j] = 1

        content = {}

        content['sentence1_token_exact_match_with_s2'] = s1_token_exact_match
        content['sentence2_token_exact_match_with_s1'] = s2_token_exact_match
        shared_content[example["pairID"]] = content
        # print(shared_content[example["pairID"]])
        # print(shared_content)


def load_nli_explain(path):
    f = open(path, "r")
    reader = csv.reader(f, delimiter=',')

    for idx, row in enumerate(reader):
        if idx ==0 : continue
        premise = row[0]
        hypothesis= row[1]
        tokens_premise = row[2].split()
        tokens_hypothesis= row[3].split()

        for t in tokens_hypothesis:
            if t.lower() not in hypothesis.lower():
                raise Exception(t)
        for t in tokens_premise:
            if t.lower() not in premise.lower():
                print(premise)
                raise Exception(t)

        yield premise, hypothesis, tokens_premise, tokens_hypothesis
