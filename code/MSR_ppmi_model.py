
# This kept crashing in jupyter, so I just made it a standalone script
# A PPMI Based model for MSR.

from nltk.corpus import wordnet
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import vsm
import data_loading
import nlu_utils

# Load the data
msr = data_loading.MSR()
dev = msr.dev()
gutenberg = msr.train_word_word_cooccurance(window=5, vocab_size=1000)


# Calculate PPMI matrix
guten_ppmi = vsm.pmi(gutenberg)


# PPMI Model
    # Features
        # Synonym Substitution with POS
    # TODO:
        # More intelligent synonym/hypernym choice
        # Sentence preprocessing

class PPMIModel:
    def __init__(self, corpus_pmi, try_synonyms=False, verbose=False):
        self.corpus_pmi = corpus_pmi
        self.index_to_label = ['a', 'b', 'c', 'd', 'e']
        self.try_synonyms = try_synonyms
        self.verbose = verbose
    
    def answer(self, problem, try_synonyms=False):
        scores = []
        question = problem['question']
        scores.append(self.score(question, problem['a)'])) 
        scores.append(self.score(question, problem['b)'])) 
        scores.append(self.score(question, problem['c)'])) 
        scores.append(self.score(question, problem['d)'])) 
        scores.append(self.score(question, problem['e)']))
        return self.index_to_label[np.argmax(scores)]
    
    def approx_ppmi(self, proposal_token, proposal_synonyms, word_token):
        pos = nlu_utils.spacy_to_wn_tag(word_token.pos_)
        word_synonyms = nlu_utils.get_alternate_words(word_token.norm_, pos)
        # First try matching using different versions of the non-proposal word
        for wsyn in word_synonyms:
            score = self.ppmi(wsyn, proposal_token.norm_)
            if score is not None:
                if self.verbose:
                    print("Used synonym: {} -> {}".format(word_token.text, wsyn))
                return score
        # Next try matching using different versions of the proposal word
        for psyn in proposal_synonyms:
            score = self.ppmi(psyn, word_token.norm_)
            if score is not None:
                if self.verbose:
                    print("Used synonym for proposal word: {} -> {}".format(proposal_token.text, psyn))
                return score
        # Next just try all combos
        for psyn in proposal_synonyms:
            for wsyn in word_synonyms:
                score = self.ppmi(psyn, word_token.norm_)
                if score is not None:
                    if self.verbose:
                        print("Used synonym: {} -> {} and {} -> {}".format(proposal_token.text, psyn, word_token.text, wsyn))
                    return score
        print("UNABLE TO FIND ANY SYNONYMS IN VOCABULARY")
        return None

    def ppmi(self, proposal, word):
        try:
            return self.corpus_pmi.loc[proposal, word]
        except KeyError:
            return None

    def substitute(self, sentence, proposal):
        sentence_list = sentence.split()
        i = sentence_list.index('_____')
        sentence_list[i] = proposal
        return ' '.join(sentence_list)

    def score(self, sentence, proposal):
        full_sentence = self.substitute(sentence, proposal)
        doc = nlu_utils.get_spacy_doc(full_sentence)
        _, proposal_token = nlu_utils.get_token(doc, proposal)

        if self.try_synonyms:
            pos = nlu_utils.spacy_to_wn_tag(proposal_token.pos_)
            synonyms = nlu_utils.get_alternate_words(proposal_token.norm_, pos)

        tot_score = 0
        for token in doc:
            if token == proposal_token:  # !!! This is dubious (might be 'is', not ==)
                continue
            if token.is_punct or token.is_space:
                continue
            score = self.ppmi(proposal_token.norm_, token.norm_)
            if score is None and self.try_synonyms:
                score = self.approx_ppmi(proposal_token, synonyms, token)
            tot_score += score if score is not None else 0
        return tot_score


# Evaluation
model = PPMIModel(guten_ppmi, try_synonyms=True)
print("Making predictions")
predictions = []
for _, problem in dev.iterrows():
    ans = model.answer(problem)
    predictions.append(ans)

print(accuracy_score(dev.loc[:, 'answer'], predictions))

# Gutenberg:
# * BASELINE: Window=5, Vocab=10000, synonyms=True: 0.482371794872
# * Window=5, Vocab=10000, synonyms_pos=True, spacy.lemmas=true: 
