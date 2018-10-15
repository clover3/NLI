



def find_ex(dictionary, target_loc, sent_tokens, pos_tags, context_tokens):



def find(dictionary, sentence):
    tokens=  tokenize(sentence)
    tags = pos_tag(tokens)
    return find_ex(dictionary, tokens, tags, [])