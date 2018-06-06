import data_loading

msr = data_loading.MSR()

windows = [5]
vocab_sizes = [12500]

for window in windows:
    for vocab_size in vocab_sizes:
        print(window, vocab_size)
        gutenberg = msr.train_word_word_cooccurence(window, vocab_size, load=False)