import data_loading

msr = data_loading.MSR()

windows = [10, 20]
vocab_sizes = [10000, 30000]

for window in windows:
    for vocab_size in vocab_sizes:
        print(window, vocab_size)
        gutenberg = msr.train_word_word_cooccurance(window, vocab_size)