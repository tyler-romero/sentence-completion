from nltk.corpus import wordnet
import spacy
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')


def treebank_to_wn_tag(tb_tag):
    tag_dict = {
        'N': 'n',
        'V': 'v',
        'J': 'a',
        'R': 'r'
    }  # R is adverb
    return tag_dict.get(tb_tag[0])


def spacy_to_wn_tag(spacy_tag):
    tag_dict = {
        'NOUN': 'n',
        'PRON': 'n',
        'PROPN': 'n',
        'VERB': 'v',
        'AUX': 'v',
        'ADJ': 'a',
        'ADV': 'r'
    }
    return tag_dict.get(spacy_tag)


def get_synonyms(word, pos=None):
    synonyms = []
    for syn in wordnet.synsets(word, pos):
        for l in syn.lemmas():
            synonyms.append(l.name())
    if not synonyms:
        print("No synonyms found for '{}' with pos={}".format(word, pos))
    return synonyms


def get_hypernyms(word, pos=None):
    # TODO: POS tagging when choosing a synset
    # TODO: find a better way to choose a synset... but there might not be that many for these words anyway
    # TODO: error handling - what do we do if a word doesn't have a synset?
    hypernym_words = []
    for syn in wordnet.synsets(word, pos):
        print("Synset: {}".format(syn))
        for h in syn.hypernyms():
            print("Hypernym synset: {}".format(h))
            for l in h.lemmas():
                hypernym_words.append(l.name())
    if not hypernym_words:
        print("No hypernyms found for '{}' with pos={}".format(word, pos))
    return hypernym_words


def get_alternate_words(word, pos=None):
    syn = get_synonyms(word, pos)
    hyp = get_hypernyms(word, pos)
    return syn + hyp


def get_spacy_doc(sentence):
    return nlp(sentence)


# Get the token of the first instance of the word in the doc
def get_token(doc, word):
    # Attempt 1
    for token in doc:
        if token.text == word:
            return token.i, token
        if token.norm_ == word:
            return token.i, token
    # Attempt 2
    word_token = nlp(word)[0]
    for token in doc:
        if token.norm_ == word_token.norm_:
            return token.i, token
    raise Exception("Unable to find word {} in spacy doc:\n{}".format(word, doc.text))


# Just use get_token
# def get_pos_of_word(spacy_doc, token_index):
#     return spacy_doc[token_index].pos_


def get_ancestors_of_word(spacy_doc, token_index):
    token = spacy_doc[token_index]
    return [ancestor for ancestor in token.ancestors]


def get_children_of_word(spacy_doc, token_index):
    token = spacy_doc[token_index]
    return [child for child in token.children]


# def remove_stop_words(sentence):
#     pass