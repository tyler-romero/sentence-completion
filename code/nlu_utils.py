from nltk.corpus import wordnet
import numpy as np
import spacy
from vsm import observed_over_expected
from collections import defaultdict

nlp = spacy.load(
    'en_core_web_sm',
    #disable=['parser', 'ner']  # Don't do dependency parsing or entity recogntion
    
    disable=['ner'] # Don't do entity recognition
)


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


def get_synonyms(word, pos=None, word_counts=None, reverse=False):
    synonyms = set()
    for syn in wordnet.synsets(word, pos):
        for l in syn.lemmas():
            synonyms.add(l.name())

    if word_counts is not None:  # Order by word_counts
        synonyms = sorted(synonyms, reverse=reverse, key=lambda x: word_counts[x])

    return list(synonyms)


def get_hypernyms(word, pos=None, word_counts=None, reverse=False):
    hypernyms = set()
    for syn in wordnet.synsets(word, pos):
        for h in syn.hypernyms():
            for l in h.lemmas():
                hypernyms.add(l.name())

    if word_counts is not None:  # Order by word_counts
        hypernyms = sorted(hypernyms, reverse=reverse, key=lambda x: word_counts[x])

    return list(hypernyms)


# Reverse = False sorts words from least frequent to most frequent
def get_alternate_words(word, pos=None, word_counts=None, reverse=False):
    syn = get_synonyms(word, pos, word_counts, reverse=reverse)
    hyp = get_hypernyms(word, pos, word_counts, reverse=reverse)
    # if not syn and not hyp:
    #     print("No alternate words found for '{}' with pos={}".format(word, pos))
    return [word] + syn + hyp


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
    raise Exception("Unable to find word '{}' in spacy doc:\n{}".format(word, doc.text))


# Just use get_token
# def get_pos_of_word(spacy_doc, token_index):
#     return spacy_doc[token_index].pos_


def get_ancestors_of_word(token, pos_to_remove=[]):
    if pos_to_remove:
        return [ancestor for ancestor in token.ancestors if ancestor.pos_ not in pos_to_remove]
    else:
        return [ancestor for ancestor in token.ancestors]
    
    

def get_children_of_word(token, pos_to_remove=[]):
    if pos_to_remove:
        return [child for child in token.children if child.pos_ not in pos_to_remove]
    else:
        return [child for child in token.children]

    
    

def dpmi(df, positive=True):
    def discount(df):
        # Calcualte mincontext
        col_totals = df.sum(axis=0)
        row_totals = df.sum(axis=1)
        mincontext = np.minimum.outer(col_totals, row_totals)

        # Calcualte discount
        discount = df / (df + 1) * mincontext / (mincontext + 1)
        return discount

    # Calculate PMI
    pmi = observed_over_expected(df)
    with np.errstate(divide='ignore'):
        pmi = np.log(pmi)
    pmi[np.isinf(pmi)] = 0.0  # log(0) = 0
    # Convert to PPMI
    if positive:
        pmi[pmi < 0] = 0.0
    # Convert to dPPMI
    pmi *= discount(df)
    return pmi




    # mincontext = np.zeros_like(df)
    # n, _ = df.shape
    # for i in range(n):
    #     for j in range(n):
    #         mincontext[i, j] = np.min([col_totals[i], row_totals[j]])