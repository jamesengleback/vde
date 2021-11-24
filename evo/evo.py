#from math import log
#import pandas as pd
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

from bm3 import BM3_DM, MXN_SITES, DOCKING_SITE
from score import score_mesotrione, mean_dists_affs

def write_json(dictionary, path, mode='a'):
    with open(path,mode) as f:
        json.dump(dictionary,f)

def write_self(path):
    with open(__name__, 'r') as f:
        code = f.read()
    with open(path,'w') as f:
        f.write(code)

def hamming(a,b):
    '''
    Hamming distance between two strings
    '''
    return sum([i != j for i,j in zip(a,b)])

def mutate_string(template, target_dict):
    '''
    mutates template string according to a dictionary {pos:aa, ... }
    '''
    s_ = list(template)
    for i,j in zip(target_dict.keys(), target_dict.values()):
        s_[i] = j
    return ''.join(s_)

def simulate(structure_template,
             sequence,
             ligand_smiles,
             binding_site,
             out_dir=None,
             keep = None,
             exhaustiveness=16,
             tmp_suffix='run',
             **kwargs
             ):
    '''
    General function for simulating a mutant
    returns enz.protein & enz.docking results
    '''
    p = enz.Protein(structure_template,
                    seq=sequence, 
                    keep=keep,
                    tmp_suffix=tmp_suffix) 

    p.refold()
    docking_results = p.dock(ligand_smiles, 
                             target_sites=binding_site,
                             exhaustiveness=exhaustiveness)
    if out_dir is not None:
        docking_results.save(out_dir)
    return p, docking_results

def evaluate(gene,
             template=None,
             exhaustiveness=16,
             out_dir=None):
    '''
    mutant evaluation function specific to this project
    '''
    assert len(gene) == len(MXN_SITES)
    out_dir = osp.join(out_dir,gene) if out_dir is not None else None
    sequence = mutate_string(BM3_DM, dict(zip(MXN_SITES, gene)))
    protein, docking_results = simulate(structure_template='4KEY.pdb',
                                        sequence=sequence,
                                        keep=['HEM'],
                                        ligand_smiles='CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]',
                                        binding_site=DOCKING_SITE,
                                        out_dir=out_dir,
                                        tmp_suffix='',
                                        exhaustiveness=exhaustiveness)
    dist_mean, aff_mean = mean_dists_affs(protein, docking_results)
    score = sum([dist_mean, aff_mean])
    if template is not None:
        ham = hamming(template, gene)
        score += ham
    else:
        ham=None
    return {'gene':gene, 
            'score':score, 
            'dist_mean':dist_mean, 
            'aff_mean':aff_mean, 
            'ham':ham}


def select_parralel(pop, 
                    parralel_fn, 
                    processes = 4, 
                    frac = 0.1,
                    out_dir=None):
    scores_dict = dict(zip(pop, parralel_fn(pop, processes)))
    df = pd.DataFrame(scores_dict).reset_index(drop=True).T
    df.columns=['score','dist_mean','aff_mean']
    df.to_csv(osp.join(out_dir, 'scores.csv'), mode = 'a', header = False)
    return  heapq.nsmallest(round(len(pop) * frac), 
                           scores_dict.keys(), 
                           key = lambda i : scores_dict[i]['score']) 
def gc(string_match):
    # garbage collection
    # can clash with other enz runs!
    files = [os.path.join('/tmp',i) for i in os.listdir('/tmp')]
    enz_files = [i for i in files if f'{string_match}_enz' in i]
    for i in enz_files:
        if os.path.isfile(i):
            os.remove(i)
        elif os.path.isdir(i):
            shutil.rmtree(i)


def main(args):
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.n_generations
    SURVIVAL = args.survival
    EXHAUSTIVENESS = args.exhaustiveness
    # ---
    RUN_ID = ''.join(random.choices(ascii_lowercase, k=5))
    OUTDIR = osp.join(args.outdir,RUN_ID)
    os.makedirs(OUTDIR)
    SCORES_CSV = osp.join(OUTDIR,'scores.csv')
    TEMPLATE = ''.join([BM3_DM[i] for i in MXN_SITES])
    # ---
    VOCAB='ACDEFGHIKLMNPQRSTVWY'
    CONFIG = {'POP_SIZE':POP_SIZE,
              'N_GENERATIONS':N_GENERATIONS,
              'SURVIVAL':SURVIVAL,
              'EXHAUSTIVENESS':EXHAUSTIVENESS,
              'VOCAB':VOCAB,
              'MXN_SITES':MXN_SITES,
              'OUTDIR':OUTDIR}

    config_path = osp.join(OUTDIR, 'config.json')
    scores_path =osp.join(OUTDIR, 'scores.json') 
    code_path =osp.join(OUTDIR, 'evo_used.py') 

    write_json(CONFIG, config_path)
    # ---

    helper = lambda gene : evaluate(gene, 
                                    exhaustiveness=EXHAUSTIVENESS,
                                    out_dir=OUTDIR)
    select_best = lambda pop, frac : heapq.nsmallest(round(len(pop) * frac), 
                                                     pop.keys(), 
                                                     key = lambda i : pop[i]['score']) 
    
    pop = [ga.random_mutate(TEMPLATE) for _ in range(POP_SIZE)] # init

    for _ in tqdm(range(N_GENERATIONS)):
        scores = ga.evaluate(pop, helper)
        write_json(scores, scores_path, 'a')
        best = select_best(scores, SURVIVAL)
        pop = [ga.crossover(*random.choices(pop, k = 2)) for i in range(POP_SIZE)]
        pop = [ga.random_mutate(i) for i in pop]
        gc(RUN_ID)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pop_size',type=int, default=10)
    parser.add_argument('-n','--n_generations',type=int, default=5)
    parser.add_argument('-e','--exhaustiveness',type=int, default=16)
    parser.add_argument('-s','--survival',type=float, default=0.25)
    parser.add_argument('-o','--outdir', default='runs')
    args = parser.parse_args()
    main(args)
