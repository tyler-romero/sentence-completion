import pandas as pd
import numpy as np
import os
import glob
import operator 
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from multithreading import run_in_parallel

data_path = '../data'
MSR_folder = 'MSR'
SAT_folder = 'SAT'
WSJ_folder = 'WSJ'

MSR_path = os.path.join(data_path, MSR_folder)
SAT_path = os.path.join(data_path, SAT_folder)
WSJ_path = os.path.join(data_path, WSJ_folder)

# TODO: Data loading for SAT and WSJ


class MSR:
    def __init__(self, path=MSR_path, seed=345):
        self.root_path = path
        self.train_path = os.path.join(path, 'Holmes_Training_Data')
        #self.test_path = os.path.join(path, 'testing_data.txt')
        self.train_files = glob.glob(os.path.join(self.train_path, '*.TXT'))
        self.seed = seed
        self.TEST_DEV_SPLIT = 0.6
        
    def vocab(self, MAX_VOCAB_SIZE):
        word_counts = defaultdict(int)
        reverse_vocab = defaultdict(lambda: None)
        
        # Word Count
        for path in tqdm(self.train_files):
            doc = self.load_document(path)
            for word in doc:
                word_counts[word] += 1
                
        # Order by count
        wc_sorted = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
        vocab = [word for word, count in wc_sorted]
        vocab = vocab[:MAX_VOCAB_SIZE]
        
        # Create reverse_vocab for fast lookup
        for i, word in enumerate(vocab):
            reverse_vocab[word] = i
            
        return vocab, reverse_vocab
        
    def load_document(self, path, verbose=False):
        with open(path) as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError:
                if verbose:
                    print("Warning: Could not load {}".format(path))
                return []
        # Remove Preamble
        for i, l in enumerate(lines):
            if '*END*THE SMALL PRINT!' in l:
                break
        lines = lines[i + 1:]
        doc = []
        for l in lines:
            l = l.split()
            doc += l
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

    # Returns a word-word co-occurance matrix
    def train_word_word_cooccurance(self, window=5, vocab_size=5000, n_threads=4):
        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)

        def word_word_coocurrance(self, doc_path):
            m = np.zeros((vocab_size, vocab_size))
            doc = self.load_document(path, verbose=False)
            for i in range(window, len(doc) - window):  # Iterate over words in document
                index_i = reverse_vocab[doc[i]]  # Get vocab index for the word in doc at index i
                if index_i is None:  # i is not in the vocab
                    continue
                for j in range(i - window, i + window + 1):  # Interate over a window for word i
                    index_j = reverse_vocab[doc[j]]
                    if i == j or index_j is None:
                        continue
                    m[index_i, index_j] += 1
                    if index_i != index_j:
                        m[index_j, index_i] += 1
            return m
        
        print('Generating matrix. n_threads={}. WARNING: This takes a VERY long time.'.format(n_threads))
        matrix = sum(run_in_parallel(word_word_coocurrance, n_threads, self.train_files))
        return pd.DataFrame(matrix, index=vocab, columns=vocab)

    def train_word_document_cooccurance(self, vocab_size=5000):
        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)
        
        print('Generating matrix.')
        # TODO: Remove docs that have a size of zero (or correctly process docs in the first place)
        matrix = np.zeros((vocab_size, len(self.train_files)))
        for doc_index, path in tqdm(enumerate(self.train_files)):
            doc = self.load_document(path, verbose=False)
            for word in doc:  # Iterate over words in document
                word_index = reverse_vocab[word]
                if word_index == None:  # if i is not in the vocab
                    continue
                matrix[word_index, doc_index] += 1
        file_names = [os.path.basename(path) for path in self.train_files]
        return pd.DataFrame(matrix, index=vocab, columns=file_names)