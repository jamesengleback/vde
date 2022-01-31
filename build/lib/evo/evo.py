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

