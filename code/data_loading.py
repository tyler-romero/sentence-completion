import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
import nltk.data
import os
import re
import glob
import operator
import string
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
from tqdm import tqdm

data_path = '../data'

MSR_path = os.path.join(data_path, 'MSR')
SAT_path = os.path.join(data_path, 'SAT')
GRE_path = os.path.join(data_path, 'GRE')
WSJ_path = os.path.join(data_path, 'WSJ')


# TODO: Data loading for SAT and WSJ
class MSR:
    def __init__(self, path=MSR_path, seed=345):
        self.root_path = path
        self.train_path = os.path.join(path, 'Holmes_Training_Data')
        #self.test_path = os.path.join(path, 'testing_data.txt')
        self.train_files = glob.glob(os.path.join(self.train_path, '*.TXT'))
        self.seed = seed
        self.TEST_DEV_SPLIT = 0.6
        self.punctuation_table = str.maketrans({key: None for key in string.punctuation})
        nlp = spacy.load('en_core_web_sm')
        self.spacy_tokenizer = Tokenizer(nlp.vocab)
        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
    def vocab(self, MAX_VOCAB_SIZE, use_spacy_norm=True):
        word_counts = self.word_count(use_spacy_norm=use_spacy_norm)
                
        # Order by count
        wc_sorted = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
        vocab = [word for word, count in wc_sorted]
        vocab = vocab[:MAX_VOCAB_SIZE]
        print("Top 10 vocab words: {}".format(vocab[:10]))
        
        # Create reverse_vocab for fast lookup
        reverse_vocab = defaultdict(lambda: None)
        for i, word in enumerate(vocab):
            reverse_vocab[word] = i
            
        return vocab, reverse_vocab

    def word_count(self, use_spacy_norm=True):
        word_counts = defaultdict(int)
        reverse_vocab = defaultdict(lambda: None)
        
        for path in tqdm(self.train_files):
            doc = self.load_document(path, split_by_sentence=(not use_spacy_norm))
            if use_spacy_norm:
                for token in doc:
                    if not token.is_space:
                        word_counts[token.norm_] += 1
            else:
                for s in doc:
                    for w in s.split():
                        word_counts[w] += 1
        return word_counts

    def load_document(self, path, verbose=False, split_by_sentence=False):
        with open(path) as f:
            full_text = f.read()
        if split_by_sentence:  # Return doc as a list of sentences
            sentences = self.nltk_tokenizer.tokenize(full_text)
            doc = [s.translate(self.punctuation_table).lower() for s in sentences]
        else:
            full_text = full_text.translate(self.punctuation_table)  # Strip punctuation
            doc = self.spacy_tokenizer(full_text)
        return doc
    
    def test(self):
        questions_path = os.path.join(self.root_path, 'testing_data.csv')
        answers_path = os.path.join(self.root_path, 'test_answer.csv')
        questions = pd.read_csv(questions_path)
        answers = pd.read_csv(answers_path)
        dataset = questions.join(answers)
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return test

    def dev(self):
        questions_path = os.path.join(self.root_path, 'testing_data.csv')
        answers_path = os.path.join(self.root_path, 'test_answer.csv')
        questions = pd.read_csv(questions_path, index_col='id')
        answers = pd.read_csv(answers_path, index_col='id')
        dataset = questions.join(answers)
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return dev

    # Returns a word-word co-occurence matrix
    # Caches each new matrix once it has been computed, to save time in the future
    def train_word_word_cooccurence(self, window=5, vocab_size=10000, decay=False, load=True, save=True):
        if decay:
            file_name = "gutenberg{}d_{}.csv.gz".format(window, vocab_size)
        else:
            file_name = "gutenberg{}_{}.csv.gz".format(window, vocab_size)
        file_path = os.path.join(self.root_path, file_name)

        # Load co-occurence matrix if it already exists
        if os.path.isfile(file_path) and load:
            print("Loading existing co-occurence matrix")
            return pd.read_csv(file_path, index_col=0, compression='gzip')

        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)
        
        print('Generating matrix')
        if decay:
            matrix = np.zeros((vocab_size, vocab_size))  # The counts will be decimals/floats
        else:
            matrix = np.zeros((vocab_size, vocab_size), dtype=np.uint32)  # The counts will be integers
        for path in tqdm(self.train_files):
            doc = self.load_document(path, verbose=False)
            for i in range(len(doc)):  # Iterate over words in document
                index_i = reverse_vocab[doc[i].norm_]  # Get vocab index for the word in doc at index i
                if index_i is None:  # i is not in the vocab
                    continue
                for j in range(i + 1, i + window + 1):  # Iterate over a FORWARD window for word i
                    if j > len(doc) - 1:  # Dont let j go out of bounds
                        continue
                    index_j = reverse_vocab[doc[j].norm_]
                    if index_j is None:  # Skip if j is not in vocab.
                        continue
                    matrix[index_i, index_j] += 1 / (j - i) if decay else 1
                    if index_i != index_j:  # Dont double count along diag
                        matrix[index_j, index_i] += 1 / (j - i) if decay else 1

        df = pd.DataFrame(matrix, index=vocab, columns=vocab)
        del matrix  # An attempt to save memory to potentially speed up saving.

        if save:  # Save co-occurence matrix
            print("Saving co-occurence matrix to {}".format(file_path))
            df.to_csv(file_path, compression='gzip')
            print("Successfully saved co-occurence matrix")
        return df

    # Word - Context (sentence) coocurence.
    def train_word_context_cooccurence(self, vocab_size=10000, load=True, save=True):
        # Load co-occurence matrix if it already exists
        file_name = "gutenberg_sent_{}.csv.gz".format(vocab_size)
        file_path = os.path.join(self.root_path, file_name)
        if os.path.isfile(file_path) and load:
            print("Loading existing co-occurence matrix")
            return pd.read_csv(file_path, index_col=0, compression='gzip')

        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size, use_spacy_norm=False)
        
        print('Generating matrix')
        matrix = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        for path in tqdm(self.train_files):
            doc = self.load_document(path, verbose=False, split_by_sentence=True)
            for sentence in doc:  # For each sentence
                sentence = sentence.split()
                for i in range(len(sentence)):  # For each word
                    index_i = reverse_vocab[sentence[i]]
                    if index_i is None:  # i is not in the vocab
                        continue
                    for j in range(i + 1, len(sentence)):  # For each other word
                        index_j = reverse_vocab[sentence[j]]
                        if index_j is None:  # Skip if j is not in vocab.
                            continue
                        matrix[index_i, index_j] += 1
                        if index_i != index_j:  # Dont double count along diag
                            matrix[index_j, index_i] += 1

        df = pd.DataFrame(matrix, index=vocab, columns=vocab)
        del matrix  # An attempt to save memory to potentially speed up saving.

        if save:  # Save co-occurence matrix
            print("Saving co-occurence matrix to {}".format(file_path))
            df.to_csv(file_path, compression='gzip')
            print("Successfully saved co-occurence matrix")
        return df

    def train_word_document_cooccurence(self, vocab_size=5000):
        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)
        
        print('Generating matrix.')
        # TODO: Remove docs that have a size of zero (or correctly process docs in the first place)
        matrix = np.zeros((vocab_size, len(self.train_files)), dtype=np.uint32)
        for doc_index, path in tqdm(enumerate(self.train_files)):
            doc = self.load_document(path, verbose=False)
            for token in doc:  # Iterate over words in document
                word_index = reverse_vocab[token.norm_]
                if word_index == None:  # if i is not in the vocab
                    continue
                matrix[word_index, doc_index] += 1
        file_names = [os.path.basename(path) for path in self.train_files]
        return pd.DataFrame(matrix, index=vocab, columns=file_names)


class GRE:
    def __init__(self, path=GRE_path, seed=345):
        self.root_path = path
        self.sentence_completion_path = os.path.join(path, 'gre_sentence_completion.json')
        self.sentence_equivalence_path = os.path.join(path, 'sentence_equivalence.json')
        self.seed = seed
        self.TEST_DEV_SPLIT = 0.6

    def dev_sentence_completion(self):
        dataset = pd.read_json(self.sentence_completion_path)
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return dev

    def test_sentence_completion(self):
        dataset = pd.read_json(self.sentence_completion_path)
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return test

    def dev_sentence_equivalence(self):
        dataset = pd.read_json(self.sentence_equivalence_path)
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return dev

    def test_sentence_equivalence(self):
        dataset = pd.read_json(self.sentence_equivalence_path)
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return test


class SAT:
    def __init__(self, path=data_path, seed=345):
        self.root_path = path

        self.sentence_completion_path = os.path.join(path, 'SAT', 'SAT-sentence-completion.json')
        self.seed = seed
        self.TEST_DEV_SPLIT = 0.6

        self.punctuation_table = str.maketrans({key: None for key in string.punctuation})
        nlp = spacy.load('en_core_web_sm')
        self.tokenizer = Tokenizer(nlp.vocab)
        
    def vocab(self, MAX_VOCAB_SIZE):
        word_counts = defaultdict(int)
        reverse_vocab = defaultdict(lambda: None)
        
        # Word Count
        for path in tqdm(self.train_files):
            doc = self.load_document(path)
            for token in doc:
                if not (token.is_punct or token.is_space):
                    word_counts[token.norm_] += 1
                
        # Order by count
        wc_sorted = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
        vocab = [word for word, count in wc_sorted]
        vocab = vocab[:MAX_VOCAB_SIZE]
        print("Top 10 vocab words: {}".format(vocab[:10]))
        
        # Create reverse_vocab for fast lookup
        for i, word in enumerate(vocab):
            reverse_vocab[word] = i
            
        return vocab, reverse_vocab

        
    def load_document(self, path, verbose=False):
        doc = []
        with open(path) as f:
            full_text = f.read()
            full_text = full_text.translate(self.punctuation_table)  # Strip punctuation
            doc = self.tokenizer(full_text)  # Use spacy
        return doc


    def dev(self):
        dataset = pd.read_json(self.sentence_completion_path)
        dataset['solution_index'] = dataset['solution_index'].apply(lambda x: x[0])
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return dev


    def test(self):
        dataset = pd.read_json(self.sentence_completion_path)
        dataset['solution_index'] = dataset['solution_index'].apply(lambda x: x[0])
        test, dev = train_test_split(dataset, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return test


    # Returns a word-word co-occurence matrix
    # Caches each new matrix once it has been computed, to save time in the future
    # TODO: Try dividing context by sentence instead of by window size!
    def train_word_word_cooccurence(self, window=5, vocab_size=15000, load=True, save=True):
        # Load co-occurence matrix if it already exists
        file_name = "nyt{}_{}.csv.gz".format(window, vocab_size)
        file_path = os.path.join(self.root_path, 'GIGA', file_name)
        if os.path.isfile(file_path) and load:
            print("Loading existing co-occurence matrix")
            return pd.read_csv(file_path, index_col=0, compression='gzip')

        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)
        
        print('Generating matrix')
        matrix = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        for path in tqdm(self.train_files):
            doc = self.load_document(path, verbose=False)
            for i in range(len(doc)):  # Iterate over words in document
                index_i = reverse_vocab[doc[i].norm_]  # Get vocab index for the word in doc at index i
                if index_i is None:  # i is not in the vocab
                    continue
                for j in range(i + 1, i + window + 1):  # Iterate over a FORWARD window for word i
                    if j > len(doc) - 1:  # Dont let j go out of bounds
                        continue
                    index_j = reverse_vocab[doc[j].norm_]
                    if index_j is None:  # Skip if j is not in vocab.
                        continue
                    matrix[index_i, index_j] += 1
                    if index_i != index_j:  # Dont double count along diag
                        matrix[index_j, index_i] += 1

        df = pd.DataFrame(matrix, index=vocab, columns=vocab)
        del matrix  # An attempt to save memory to potentially speed up saving.

        if save:  # Save co-occurence matrix
            print("Saving co-occurence matrix to {}".format(file_path))
            df.to_csv(file_path, compression='gzip')
            print("Successfully saved co-occurence matrix")
        return df





class GIGA:
    def __init__(self, path=data_path, seed=345):
        self.root_path = path
        # self.train_path = os.path.join(path, '.') # TODO change train path to 'nyt_eng'
        #self.test_path = os.path.join(path, 'testing_data.txt')
        # self.train_files = glob.glob(os.path.join(self.train_path, '*')) # TODO uncomment
        self.train_files = glob.glob(os.path.join(self.root_path, '*.txt'))
        self.seed = seed
        self.TEST_DEV_SPLIT = 0.6
        self.punctuation_table = str.maketrans({key: None for key in string.punctuation if key != '-'})
        nlp = spacy.load('en_core_web_sm')
        self.tokenizer = Tokenizer(nlp.vocab)
        
    def vocab(self, MAX_VOCAB_SIZE):
        word_counts = defaultdict(int)
        reverse_vocab = defaultdict(lambda: None)
        
        # Word Count
        for path in tqdm(self.train_files):
            doc = self.load_document(path)
            for token in doc:
                if not (token.is_punct or token.is_space):
                    word_counts[token.norm_] += 1
                
        # Order by count
        wc_sorted = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
        vocab = [word for word, count in wc_sorted]
        vocab = vocab[:MAX_VOCAB_SIZE]
        print("Top 10 vocab words: {}".format(vocab[:10]))
        
        # Create reverse_vocab for fast lookup
        for i, word in enumerate(vocab):
            reverse_vocab[word] = i
            
        return vocab, reverse_vocab
        
    def load_document(self, path, verbose=False):
        doc = []
        with open(path) as f:
            full_text = f.read()

            # Remove all markdown tags and whatever is in dateline, if it exists
            full_text = re.sub('<DATELINE>\n.+\n</DATELINE>', '', full_text)
            full_text = re.sub('<.?DOC.*>', '', full_text)
            full_text = re.sub('<.?HEADLINE>', '', full_text)
            full_text = re.sub('<.?TEXT>', '', full_text)
            full_text = re.sub('<.?P>', '', full_text)


            full_text = full_text.translate(self.punctuation_table)  # Strip punctuation

            print (full_text)

            doc = self.tokenizer(full_text)  # Use spacy
        return doc


    # Returns a word-word co-occurence matrix
    # Caches each new matrix once it has been computed, to save time in the future
    # TODO: Try dividing context by sentence instead of by window size!
    def train_word_word_cooccurence(self, name='giga', window=5, vocab_size=10000, load=True, save=True):
        # Load co-occurence matrix if it already exists
        file_name = "{}{}_{}.csv.gz".format(name, window, vocab_size)
        file_path = os.path.join(self.root_path, file_name)
        if os.path.isfile(file_path) and load:
            print("Loading existing co-occurence matrix")
            return pd.read_csv(file_path, index_col=0, compression='gzip')

        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)
        
        print('Generating matrix')
        matrix = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        for path in tqdm(self.train_files):
            doc = self.load_document(path, verbose=False)
            for i in range(len(doc)):  # Iterate over words in document
                index_i = reverse_vocab[doc[i].norm_]  # Get vocab index for the word in doc at index i
                if index_i is None:  # i is not in the vocab
                    continue
                for j in range(i + 1, i + window + 1):  # Iterate over a FORWARD window for word i
                    if j > len(doc) - 1:  # Dont let j go out of bounds
                        continue
                    index_j = reverse_vocab[doc[j].norm_]
                    if index_j is None:  # Skip if j is not in vocab.
                        continue
                    matrix[index_i, index_j] += 1
                    if index_i != index_j:  # Dont double count along diag
                        matrix[index_j, index_i] += 1

        df = pd.DataFrame(matrix, index=vocab, columns=vocab)
        del matrix  # An attempt to save memory to potentially speed up saving.

        if save:  # Save co-occurence matrix
            print("Saving co-occurence matrix to {}".format(file_path))
            df.to_csv(file_path, compression='gzip')
            print("Successfully saved co-occurence matrix")
        return df




