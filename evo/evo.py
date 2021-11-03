from math import log
import pandas as pd
import os
import os.path as osp
import shutil
import json
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

RUN_ID = ''.join(random.choices(ascii_lowercase, k=5))
OUTDIR = osp.join('runs','newscore',RUN_ID)
WT = ''.join([BM3_WT[i] for i in MXN_SITES])
K = 0.5 # hamming dist correction term


def hamming(a,b):
    return sum([1 for i,j in zip(a,b) if i != j])

def evaluate(gene,
             exhaustiveness=16,
             out_dir=OUTDIR):
    mutation_dictionary = dict(zip(MXN_SITES, gene))
    p = enz.protein('../data/4KEY.pdb',
                    seq = BM3_WT, # my residue numbering system
                    keep = ['HEM'],
                    tmp_suffix=RUN_ID) # keep the heme
    for pos, aa in zip(mutation_dictionary.keys(), 
                       mutation_dictionary.values()):
        p.mutate(pos, aa)
    p.refold()
    docking_results = p.dock('CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]', 
                             target_sites = DOCKING_SITE,
                             exhaustiveness = exhaustiveness)
    docking_results.save(osp.join(out_dir, gene))
    score_m, dist_mean, aff_mean = score_mesotrione(p, docking_results)  # todo return dist, aff, score
    ham = hamming(WT, gene)
    score = score_m + (0.1 * log(1 + ham))
    return {'gene':gene, 'score':score, 'dist_mean':dist_mean, 'aff_mean':aff_mean, 'ham':ham}


def select_parralel(pop, 
                    parralel_fn, 
                    processes = 4, 
                    frac = 0.1,
                    out_dir=OUTDIR):
    scores_dict = dict(zip(pop, parralel_fn(pop, processes)))
    df = pd.DataFrame(scores_dict).reset_index(drop=True).T
    df.columns=['score','dist_mean','aff_mean']
    df.to_csv(osp.join(out_dir, 'scores.csv'), mode = 'a', header = False)
    return  heapq.nsmallest(round(len(pop) * frac), 
                           scores_dict.keys(), 
                           key = lambda i : scores_dict[i]['score']) 
def gc():
    # garbage collection
    # can clash with other enz runs!
    files = [os.path.join('/tmp',i) for i in os.listdir('/tmp')]
    enz_files = [i for i in files if f'{RUN_ID}_enz' in i]
    for i in enz_files:
        if os.path.isfile(i):
            os.remove(i)
        elif os.path.isdir(i):
            shutil.rmtree(i)


def main(args):
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.n_generations
    SURVIVAL = args.survival
    os.makedirs(OUTDIR)
    SCORES_CSV = osp.join(OUTDIR,'scores.csv')
    WT = ''.join([BM3_WT[i] for i in MXN_SITES])
    VOCAB='ACDEFGHIKLMNPQRSTVWY'
    EXHAUSTIVENESS = args.exhaustiveness
    # write run config
    CONFIG = {'POP_SIZE':POP_SIZE,
              'N_GENERATIONS':N_GENERATIONS,
              'SURVIVAL':SURVIVAL,
              'EXHAUSTIVENESS':EXHAUSTIVENESS,
              'VOCAB':VOCAB,
              'MXN_SITES':MXN_SITES,
              'OUTDIR':OUTDIR}

    with open(osp.join(OUTDIR, 'config.json'),'w') as f:
        json.dump(CONFIG,f)


    pd.DataFrame([], columns=['gene','score','dist_mean','aff_mean','ham'])\
            .to_csv(osp.join(OUTDIR, 'scores.csv'),index=False)

    helper = lambda gene : evaluate(gene, exhaustiveness=EXHAUSTIVENESS)
    select_best = lambda pop, frac : heapq.nsmallest(round(len(pop) * frac), 
                                               scores_dict.keys(), 
                                               key = lambda i : scores_dict[i]['score']) 
    
    pop = [ga.mutate(WT) for i in range(POP_SIZE)]
    for i in tqdm(range(N_GENERATIONS)):
        scores = ga.eval(pop, helper)
        print(scores)

        pd.DataFrame(scores).T.to_csv(osp.join(OUTDIR, 'scores.csv'), header=False, index=False, mode='a')

        best = heapq.nsmallest(round(SURVIVAL * POP_SIZE),
                                   scores.keys(),
                                   key = lambda i : scores[i]['score']) ## heapq lookup

        pop = [ga.crossover(*random.choices(pop, k = 2)) for i in range(POP_SIZE)]
        pop = [ga.mutate(i) for i in pop]
        gc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pop_size',type=int, default=10)
    parser.add_argument('-n','--n_generations',type=int, default=5)
    parser.add_argument('-e','--exhaustiveness',type=int, default=16)
    parser.add_argument('-s','--survival',type=float, default=0.25)
    args = parser.parse_args()
    main(args)
