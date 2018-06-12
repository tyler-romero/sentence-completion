import data_loading

msr = data_loading.MSR()

windows = [20]
vocab_sizes = [10000]

for window in windows:
    for vocab_size in vocab_sizes:
        print("Word word coocurrence: ", window, vocab_size)
        gutenberg = msr.train_word_word_cooccurence(window, vocab_size, decay=True, load=False)

# for vocab_size in vocab_sizes:
#         print("Word context coocurrence: ", vocab_size)
#         gutenberg = msr.train_word_context_cooccurence(vocab_size, load=False)