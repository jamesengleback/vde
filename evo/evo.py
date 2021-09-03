import pandas as pd
import os
import osp as osp
import random
from string import ascii_lowercase
import heapq
from multiprocessing.pool import ThreadPool
import numpy as np
from tqdm import tqdm
import argparse

import enz
import ga

from bm3 import BM3_WT, MXN_SITES, DOCKING_SITE
from score import score_mesotrione 

ID = ''.join(random.choices(ascii_lowercase, k=5))
OUTPATH = osp.join('..','runs', ID)
WT = ''.join([BM3_WT[i] for i in ACTIVE_SITE_AAS])
K = 0.5 # hamming dist correction term


def hamming(a,b):
    return sum([1 for i,j in zip(a,b) if i != j])

def evaluate(gene,
             exhaustiveness=16):
    mutation_dictionary = dict(zip(ACTIVE_SITE_AAS, gene))
    p = enz.protein('../data/4KEY.pdb',
                    seq = BM3_WT, # my residue numbering system
                    cofactors = ['HEM']) # keep the heme
    for pos, aa in zip(mutation_dictionary.keys(), 
                       mutation_dictionary.values()):
        p.mutate(pos, aa)
    p.refold()
    docking_results = p.dock('CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]', 
                             target_residues = DOCKING_SITE,
                             exhaustiveness = exhaustiveness)
    docking_results.save(osp.join(OUTPATH, gene))
    score_m, dist_mean, aff_mean = score_mesotrione(p, docking_results)  # todo return dist, aff, score
    ham = hamming(WT, gene)
    score = score_m / ham 
    return {'score':score, 'dist_mean':dist_mean, 'aff_mean':aff_mean}


def select_parralel(pop, 
                    parralel_fn, 
                    processes = 4, 
                    frac = 0.1):
    scores_dict = dict(zip(pop, parralel_fn(pop, processes)))
    df = pd.DataFrame(scores_dict).reset_index(drop=True).T
    df.columns=['score','dist_mean','aff_mean']
    df.to_csv(osp.join(OUTPATH, 'scores.csv'), mode = 'a', header = False)
    return  heapq.nsmallest(round(len(pop) * frac), 
                           scores_dict.keys(), 
                           key = lambda i : scores_dict[i]['score']) 

def main(args):
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.n_generations
    SURVIVAL = args.survival
    os.makedirs(RUN_ID)
    SCORES_CSV = osp.join(RUN_ID,'scores.csv')
    #MXN_SITES = [130, 197, 223, 224, 225, 280, 308, 310, 343] # targetting to hydrophobic
    WT = ''.join([BM3_WT[i] for i in ACTIVE_SITE_AAS])
    VOCAB='ACDEFGHIKLMNPQRSTVWY'
    EXHAUSTIVENESS = args.exhaustiveness
    # write run config
    CONFIG = {'POP_SIZE':POP_SIZE,
              'N_GENERATIONS':N_GENERATIONS,
              'SURVIVAL':SURVIVAL,
              'PE_N_MER':PE_N_MER,
              'EXHAUSTIVENESS':EXHAUSTIVENESS,
              'VOCAB':VOCAB,
              'MXN_SITES':MXN_SITES,
              'RUN_ID':RUN_ID}

    with open(osp.join(RUN_ID, 'config.json'),'w') as f:
        json.dump(CONFIG,f)


    os.mkdir(OUTPATH)
    pd.DataFrame([], columns=['score','dist_mean','aff_mean']).to_csv(osp.join(OUTPATH, 'scores.csv'))

    helper = lambda gene : evaluate(gene, exhaustiveness=EXHAUSTIVENESS)
    select_best = lambda pop, frac : heapq.nsmallest(round(len(pop) * frac), 
                                               scores_dict.keys(), 
                                               key = lambda i : scores_dict[i]['score']) 
    
    pop = [mutate(WT) for i in range(POP_SIZE)]
    for i in tqdm(range(N_GENERATIONS)):
        scores = ga.eval(pop, helper)
        print(scores)
        pop = [ga.crossover(*random.choices(pop, k = 2)) for i in range(pop_size)]
        pop = [ga.mutate(i) for i in pop]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pop_size',type=int, default=10)
    parser.add_argument('-n','--n_generations',type=int, default=5)
    parser.add_argument('-m','--pe_n_mer',type=int, default=50)
    parser.add_argument('-e','--exhaustiveness',type=int, default=16)
    parser.add_argument('-s','--survival',type=float, default=0.25)
    args = parser.parse_args()
    main(args)
