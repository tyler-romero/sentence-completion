from multiprocessing import Pool

def run_in_parallel(func, n_threads, work):
    # work is a list of calls (arguements) to func that need to be completed
    p = Pool(n_threads)
    return p.map(func, args)  # Returns [func(work[0]), func(work[1]), ..., func(work[n])]