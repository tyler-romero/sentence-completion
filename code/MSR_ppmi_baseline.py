
# This kept crashing in jupyter, so I just made it a standalone script

# A Baseline for the MSR dataset using gigaword and PPMI

from nltk.corpus import wordnet
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import vsm
import data_loading


# Load the data

msr = data_loading.MSR()
dev = msr.dev()
gutenberg = msr.train_word_word_cooccurance(window=5, vocab_size=10000)

#vsmdata = '../data/vsmdata'
#giga5 = pd.read_csv(os.path.join(vsmdata, 'gigaword_window5-scaled.csv.gz'), index_col=0)


# Calculate PPMI matrix

# giga5_pmi = vsm.pmi(giga5)
guten_ppmi = vsm.pmi(gutenberg)


# ## PPMI Model
# From Inkpen 2007

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms


class PPMIBaseline:
    def __init__(self, corpus_pmi, try_synonyms=False):
        self.corpus_pmi = corpus_pmi
        self.index_to_label = ['a', 'b', 'c', 'd', 'e']
        self.try_synonyms = try_synonyms
    
    def answer(self, problem, try_synonyms=False):
        scores = []
        question = problem['question']
        scores.append(self.score(question, problem['a)'])) 
        scores.append(self.score(question, problem['b)'])) 
        scores.append(self.score(question, problem['c)'])) 
        scores.append(self.score(question, problem['d)'])) 
        scores.append(self.score(question, problem['e)']))
        return self.index_to_label[np.argmax(scores)]
    
    def ppmi(self, proposal, word):
        try:
            return self.corpus_pmi.loc[proposal, word]
        except KeyError:
            return None
    
    def score(self, sentence, proposal):
        sentence = sentence.lower()
        score = 0
        if self.try_synonyms:
            synonyms = get_synonyms(proposal)
        for word in sentence.split():
            if word == '_____':
                continue
            s = self.ppmi(proposal, word)
            if s is None and self.try_synonyms:
                for syn in synonyms:
                    s = self.ppmi(syn, word)
                    if s is not None:
                        break 
            score += s if s is not None else 0
        return score


# ## Evaluation

model = PPMIBaseline(guten_ppmi, try_synonyms=True)
print("Making predictions")
predictions = []
for _, problem in dev.iterrows():
    ans = model.answer(problem)
    predictions.append(ans)

print(accuracy_score(dev.loc[:, 'answer'], predictions))


# Note: part of the reason this does so poorly is that the majority of the answers are not even in the gigaword vocabulary.
# * Before adding synonym matching (giga20): 0.27884615384615385
# * After adding synonym matching (giga20): 0.26442307692307693    :(
# 
# Thoughts on why:
# Using synonyms gives more non-zero scores, but doesn't necessarily give more nonzero scores to the correct answer categories. Just the words that have common synonyms.
# 
# Gutenberg:
# * Window=5, Vocab=5000, synonyms=False: 0.36538461538461536 (before I decided to strip punctuation)
# * Window=5, Vocab=10000, synonyms=False: 0.44871794871794873 
# * Window=5, Vocab=10000, synonyms=True: 0.482371794872
# * Window=10, Vocab=10000, synonyms=True: 0.464743589744
# * Window=10, Vocab=10000, synonyms=Flase: 0.442307692308
