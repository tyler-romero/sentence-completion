import pandas as pd
import numpy as np
import os
import glob
import operator 
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

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
        self.test_path = os.path.join(path, 'testing_data.txt')
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
    
    def parse_question(self, line):
        question_number = line.split()[0]
        answer = line.split()[1:]
        for i, c in enumerate(question_number):
            if c.isalpha():
                number = int(question_number[:i])
                letter = question_number[i]
                return number, answer
        raise IOError

    def read_questions(self):
        test = []
        # Read in as a list of lines
        with open(self.test_path) as f:
            test_lines = f.readlines()
        test_lines = [x.strip() for x in test_lines]

        # Process data into a usable test set
        current = 1
        ex = []
        for line in test_lines:
            number, answer = self.parse_question(line)
            if current != number:
                test.append(ex)
                ex = []
            current = number
            ex.append(answer)
        test.append(ex)
        return test
    
    def dev(self):
        questions = self.read_questions()
        test, dev = train_test_split(questions, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return dev
    
    def test(self):
        questions = self.read_questions()
        test, dev = train_test_split(questions, test_size=self.TEST_DEV_SPLIT, random_state=self.seed)
        return test
    
    # Returns a word-word co-occurance matrix
    def train_word_word_cooccurance(self, window=5, vocab_size=5000):
        print('Loading vocab')
        vocab, reverse_vocab = self.vocab(vocab_size)
        
        print('Generating matrix. WARNING: This takes a VERY long time.')
        matrix = np.zeros((vocab_size, vocab_size))
        for path in tqdm(self.train_files):
            doc = self.load_document(path, verbose=False)
            for i in range(window, len(doc) - window):  # Iterate over words in document
                index_i = reverse_vocab[doc[i]]  # Get vocab index for the word in doc at index i
                if index_i == None:  # i is not in the vocab
                    continue
                for j in range(i - window, i + window + 1):  # Interate over a window for word i
                    index_j = reverse_vocab[doc[j]]
                    if i == j or index_j == None:
                        continue
                    matrix[index_i, index_j] += 1
                    if index_i != index_j:
                        matrix[index_j, index_i] += 1
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