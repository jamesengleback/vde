import random
import heapq
import json
#from multiprocessing.pool import ThreadPool, Pool
from concurrent.futures import ThreadPoolExecutor#, ProcessPoolExecutor

AAS = list('ACDEFGHIKLMNPQRSTVWY')

def random_seq(n : int, 
               vocab=AAS):
    ''' generates random string of length n from characters in vocab 
        (iterable returning strings)
    '''
    return ''.join(random.choices(vocab,k=n))

def mutate(seq, 
           pos:int, 
           new:str):
    ''' mutate string at pos to new
    '''
    seq = list(seq)
    seq[pos] = new
    return ''.join(seq)

def hamming(a:str,
            b:str):
    ''' return hamming distance between two strings of the same length
    '''
    assert len(a) == len(b)
    return sum([i!=j for i,j in zip(a,b)])

def random_mutate(seq:str, 
                  vocab=AAS, 
                  pos_weights=None, 
                  vocab_weights=None):
    ''' mutate string at random position to random characters
        from vocab (iterable of chars),
        seq : str 
        vocab : iterable returning strings
        pos_weights : iterable of floats, maps to seq
            probability weights for position selection
        vocab_weights : iterable of floats, maps to vocab 
            probability weights for substitution selection
    '''
    mxn_site = random.choices(range(len(seq)), weights=pos_weights, k=1)[0]
    new = random.choices(vocab, weights=vocab_weights, k=1)[0]
    return mutate(seq, mxn_site, new)

def crossover(a:str,
              b:str):
    ''' randomly splice two strings
        returns string
    '''
    cut = random.randint(1,min(len(a),len(b))-1)
    return random.choice([a[:cut] + b[cut:], b[:cut] + a[cut:]])

def evaluate(gene_pool, 
             fn, 
             **kwargs):
    ''' gene_pool : iterable (except generators)
        fn : function to map to gene_pool
        map a function to an iterable with multiprocessing 
        return original iterable (?generators) and list of 
        function evaluations

    '''
    #fn = lambda x : fn_(x,  **kwargs)
    with ThreadPoolExecutor(**kwargs) as process_pool :
        results = process_pool.map(fn, gene_pool)
    return gene_pool, list(results)

