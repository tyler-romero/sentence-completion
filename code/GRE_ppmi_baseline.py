
# coding: utf-8

# In[ ]:

import numpy as np
import os
import pandas as pd
import vsm
import data_loading
import itertools

# from nltk.corpus import wordnet


# In[2]:

gre = data_loading.GRE()


# In[3]:

dev = gre.dev_sentence_completion()
print(dev.shape)
dev.head()


# In[ ]:

msr = data_loading.MSR()
gutenberg = msr.train_word_word_cooccurance(window=5, vocab_size=30000)
guten_ppmi = vsm.pmi(gutenberg)


# In[ ]:

dev["question"][5]


# In[ ]:

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms


class PPMIBaseline:
    def __init__(self, corpus_pmi):
        self.corpus_pmi = corpus_pmi
    
    def answer(self, problem):
        n_blanks = problem['num_blanks']
        if n_blanks == 1:
            return self.answer1(problem)
        elif n_blanks == 2:
            return self.answer2(problem)
        else:
            return self.answer3(problem)
    
    def answer1(self, problem):
        scores = []
        for option1 in problem["candidates"]:
            scores += [self.score1(problem["question"], option1)]
        return [np.argmax(scores)]
        
    def answer2(self, problem):
        scores = []
        combos = list(itertools.product(problem["candidates"], problem["candidates_2"]))
        for option1, option2 in combos:
            scores += [self.score2(problem["question"], option1, option2)]
        ans1, ans2 = combos[np.argmax(scores)]
        return [problem["candidates"].index(ans1), problem["candidates_2"].index(ans2)]
        
    def answer3(self, problem):
        scores = []
        combos = list(itertools.product(problem["candidates"], problem["candidates_2"], problem["candidates_3"]))
        for option1, option2, option3 in combos:
            scores += [self.score3(problem["question"], option1, option2, option3)]
        ans1, ans2, ans3 = combos[np.argmax(scores)]
        return [problem["candidates"].index(ans1), problem["candidates_2"].index(ans2), problem["candidates_3"].index(ans3)]
            
    def ppmi(self, proposal, word):
        try:
            return self.corpus_pmi.loc[proposal, word]
        except KeyError:
            return 0
        
    def score1(self, sentence, option1):
        sentence = sentence.lower()
        score = 0
        for word in sentence.split():
            if word == '$BLANK_0':
                continue
            score += self.ppmi(option1, word)
        return score
        
    def score2(self, sentence, option1, option2):
        sentence = sentence.lower()
        score = 0
        for word in sentence.split():
            if word == '$BLANK_0' or '$BLANK_1':
                continue
            score += self.ppmi(option1, word)
            score += self.ppmi(option2, word)
        score += self.ppmi(option1, option2)
        return score
    
    def score3(self, sentence, option1, option2, option3):
        sentence = sentence.lower()
        score = 0
        for word in sentence.split():
            if word == '$BLANK_0' or '$BLANK_1' or '$BLANK_2':
                continue
            score += self.ppmi(option1, word)
            score += self.ppmi(option2, word)
            score += self.ppmi(option3, word)
        score += self.ppmi(option1, option2)
        score += self.ppmi(option1, option3)
        score += self.ppmi(option2, option3)
        return score


# In[ ]:

model = PPMIBaseline(guten_ppmi)
predictions = []
for _, problem in dev.iterrows():
    ans = model.answer(problem)
    predictions.append(ans)


# In[ ]:

def accuracy_score(predictions, dev):
    n_correct = 0.0
    for i, (_, problem) in enumerate(dev.iterrows()):
        print(problem["solution_index"], "or", problem["solution_indices"], "==", predictions[i])
        if problem["solution_index"] == predictions[i] or problem["solution_indices"] == predictions[i]:
            n_correct += 1
    return n_correct / len(predictions)
        
accuracy_score(predictions, dev)


# In[ ]:




# In[ ]:



