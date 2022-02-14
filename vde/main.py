import os
import os.path as osp
import random
from string import ascii_lowercase
import json
from tqdm import tqdm
import argparse
import shutil
import pandas as pd

import enz

import ga
import utils
from utils import BM3_DM, MXN_SITES, DOCKING_SITE
from sfxns import score_a, score_b, score_c, mean_dists_affs

def evaluate(gene,
             out_dir_root,
             score_fn,
             template=None,
             exhaustiveness=16,
             run_id=None,
             ):
    assert len(gene) == len(MXN_SITES)
    sequence = utils.mutate_string(BM3_DM, dict(zip(MXN_SITES, gene)))
    protein = enz.Protein('4KEY.pdb',
                          seq=sequence, 
                          keep=['HEM'],
                          tmp_suffix=run_id) 
    protein.refold()
    docking_results = protein.dock('CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]',
                                   target_sites=DOCKING_SITE,
                                   exhaustiveness=exhaustiveness)
    dist_mean, aff_mean, score = score_fn(protein, docking_results)
    out_dir = osp.join(out_dir_root,gene) 
    docking_results.save(out_dir)
    ham = ga.hamming(template, gene)
    score += ham
    return {'gene':gene, 
            'score':score, 
            'dist_mean':dist_mean, 
            'aff_mean':aff_mean, 
            'ham':ham,
            }


def main(args):
    # --- config stuff ---
    POP_SIZE = args.pop_size
    N_GENERATIONS = args.n_generations
    SURVIVAL = args.survival
    EXHAUSTIVENESS = args.exhaustiveness
    SCOREFN = args.score_fn
    assert SCOREFN in {'a','b','c'}
    RUN_ID = ''.join(random.choices(ascii_lowercase, k=5))
    OUTDIR = osp.join(args.outdir,RUN_ID)
    os.makedirs(OUTDIR)
    SCORES_CSV = osp.join(OUTDIR,'scores.csv')
    TEMPLATE = ''.join([BM3_DM[i] for i in MXN_SITES])
    VOCAB='ACDEFGHIKLMNPQRSTVWY'
    CONFIG = {'POP_SIZE':POP_SIZE,
              'N_GENERATIONS':N_GENERATIONS,
              'SURVIVAL':SURVIVAL,
              'EXHAUSTIVENESS':EXHAUSTIVENESS,
              'VOCAB':VOCAB,
              'MXN_SITES':MXN_SITES,
              'OUTDIR':OUTDIR, 
              'SCOREFN':SCOREFN}
    config_path = osp.join(OUTDIR, 'config.json')
    scores_path =osp.join(OUTDIR, 'scores.json') 
    code_path =osp.join(OUTDIR, 'evo_used.py') 
    utils.write_json(CONFIG, config_path)
    score_fn = {'a':score_a, 'c':score_c, 'c':score_c}[SCOREFN]
    # ---
    def helper(gene):
        output = evaluate(gene, 
                          exhaustiveness=EXHAUSTIVENESS,
                          template=TEMPLATE,
                          out_dir_root=OUTDIR,
                          run_id=RUN_ID,
                          score_fn=score_fn)
        uid = ''.join(random.choices(ascii_lowercase, k=5))
        output['uid'] = uid
        scores_path = osp.join(OUTDIR, 'scores.csv')
        utils.write_csv(output, scores_path)
        return output['score']

    pop = [ga.random_mutate(TEMPLATE) for _ in range(POP_SIZE)] # init

    for _ in tqdm(range(N_GENERATIONS)):
        pop, scores = ga.evaluate(helper, pop)
        #scores_ = ga.evaluate(gene_pool, eval_helper)
        #scores = dict(zip(*scores_))
        best = heapq.nsmallest(round(SURVIVAL * POP_SIZE),
                                scores.keys(),
                                key = lambda i : scores[i]['score'])
        pop = [ga.crossover(*random.choices(best,k=2)) for _ in range(POP_SIZE)]
        pop = [ga.mutate(i) for i in pop]

        out = pd.DataFrame(scores).T
        out['gene'] =  gene_pool
        out.to_csv(scores_csv, mode='a', header=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pop_size',type=int, default=8)
    parser.add_argument('-n','--n_generations',type=int, default=5)
    parser.add_argument('-e','--exhaustiveness',type=int, default=1)
    parser.add_argument('-s','--survival',type=float, default=0.25)
    parser.add_argument('-o','--outdir', default='runs')
    parser.add_argument('-fn','--score_fn', default='a')
    args = parser.parse_args()
    main(args)
