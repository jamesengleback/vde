# todo - save structures


import enz
import os
import random
from string import ascii_lowercase
import heapq
import multiprocessing
import numpy as np
from tqdm import tqdm
import argparse

from bm3 import BM3_DM, ACTIVE_SITE_AAS
from score import score_mesotrione 

ID = ''.join(random.choices(ascii_lowercase, k=5))

def cross(a, b):
    cut_point = random.randint(0, min(len(a), len(b)))
    ab = a[:cut_point] + b[cut_point:]
    ba = b[:cut_point] + a[cut_point:]
    return random.choice([ab,ba])

def mutate(gene):
    gene = list(gene) # lists can be modified
    pos = random.randint(0, len(gene) - 1)
    gene[pos] = random.choice('ACDEFGHIKLMNPQRSTVWY')
    return ''.join(gene)

def evaluate(gene):
    mutation_dictionary = dict(zip(ACTIVE_SITE_AAS, gene))
    p = enz.protein('../data/4KEY.pdb',
                    seq = BM3_DM, # my residue numbering system
                    cofactors = ['HEM']) # keep the heme

    for pos in mutation_dictionary:
        aa = mutation_dictionary[pos]
        p.mutate(pos, aa)
    p.refold()

    docking_results = p.dock('CCS(=O)(=O)C1=C(N2C=CC=CC2=N1)S(=O)(=O)NC(=O)NC3=NC(=CC(=N3)OC)OC',
                             target_residues = ACTIVE_SITE_AAS,
                             exhaustiveness = 1)
    docking_results.save(os.path.join(savedir, gene))
    return score(p, docking_results) 

def evaluate_batch(pop):
    with multiprocessing.Pool(len(pop)) as pool:
        results = pool.map(evaluate, pop)
    pool.join()
    return results

def select_parralel(pop, parralel_fn, frac = 0.1):
    scores_dict = dict(zip(pop, parralel_fn(pop)))
    return  heapq.nsmallest(round(len(pop) * frac), 
                           scores_dict.keys(), 
                           key = lambda i : scores_dict[i]) 

def main(args):
    pop_size = args.popsize
    if args.id != None:
        global ID 
        ID = args.id

    os.mkdir(ID)
    
    pop = [mutate('TYLFVLLIA') for i in range(pop_size)]
    for i in tqdm(range(args.nrounds)):
        pop = select_parralel(pop, evaluate_batch, frac = 0.25)
        pop = [cross(*random.choices(pop, k = 2)) for i in range(pop_size)]
        pop = [mutate(i) for i in pop]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nrounds', default = 3)
    parser.add_argument('-p', '--popsize', default = 8)
    parser.add_argument('-i','--id')
    args = parser.parse_args()
    main(args)
