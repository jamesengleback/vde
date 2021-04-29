import pandas as pd
import enz
import os
import random
from string import ascii_lowercase
import heapq
import multiprocessing
import numpy as np
from tqdm import tqdm
import argparse

from bm3 import BM3_WT, ACTIVE_SITE_AAS, DOCKING_SITE
from score import score_mesotrione 

ID = ''.join(random.choices(ascii_lowercase, k=5))
OUTPATH = os.path.join('..','runs', ID)
WT = ''.join([BM3_WT[i] for i in ACTIVE_SITE_AAS])
K = 0.5 # hamming dist correction term

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

def hamming(a,b):
    return sum([1 for i,j in zip(a,b) if i != j])

def evaluate(gene):
    mutation_dictionary = dict(zip(ACTIVE_SITE_AAS, gene))
    p = enz.protein('../data/4KEY.pdb',
                    seq = BM3_WT, # my residue numbering system
                    cofactors = ['HEM']) # keep the heme

    for pos in mutation_dictionary:
        aa = mutation_dictionary[pos]
        p.mutate(pos, aa)
    p.refold()

    docking_results = p.dock('CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]', 
                             target_residues = DOCKING_SITE,
                             exhaustiveness = 16)
    docking_results.save(os.path.join(OUTPATH, gene))
    score_m, dist_mean, aff_mean = score_mesotrione(p, docking_results)  # todo return dist, aff, score
    ham = hamming(WT, gene)
    score = score_m / ham 
    return {'score':score, 'dist_mean':dist_mean, 'aff_mean':aff_mean}

def evaluate_batch(pop, processes):
    with multiprocessing.Pool(processes = processes) as pool:
        results = pool.map(evaluate, pop)
    pool.join()
    return results

def select_parralel(pop, parralel_fn, processes = 4, frac = 0.1):
    scores_dict = dict(zip(pop, parralel_fn(pop, processes)))
    df = pd.DataFrame(scores_dict).reset_index(drop=True).T
    df.columns=['score','dist_mean','aff_mean']
    print(df)
    print(scores_dict)
    df.to_csv(os.path.join(OUTPATH, 'scores.csv'), mode = 'a', header = False)
    return  heapq.nsmallest(round(len(pop) * frac), 
                           scores_dict.keys(), 
                           key = lambda i : scores_dict[i]['score']) 

def main(args):
    pop_size = args.popsize
    if args.id != None:
        global ID 
        ID = args.id

    os.mkdir(OUTPATH)
    pd.DataFrame([], columns=['score','dist_mean','aff_mean']).to_csv(os.path.join(OUTPATH, 'scores.csv'))
    
    pop = [mutate(WT) for i in range(pop_size)]
    for i in tqdm(range(args.nrounds)):
        pop = select_parralel(pop, evaluate_batch, processes = args.cpus, frac = 0.25)
        pop = [cross(*random.choices(pop, k = 2)) for i in range(pop_size)]
        pop = [mutate(i) for i in pop]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nrounds', default = 3, type = int)
    parser.add_argument('-p', '--popsize', default = 8, type = int)
    parser.add_argument('-i','--id')
    parser.add_argument('-c','--cpus', default = 4, type = int)
    args = parser.parse_args()
    main(args)
