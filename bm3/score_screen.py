import os
import os.path as osp
import random
from string import ascii_lowercase
import argparse

import ga
import evo

import score
from bm3 import BM3_DM, MXN_SITES, DOCKING_SITE
from main import evaluate

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
    evo.write_json(CONFIG, config_path)

    def helper(gene, **kwargs):
        # crashes still happen sometimes :(
        try:
            output = evaluate(gene, 
                              exhaustiveness=EXHAUSTIVENESS,
                              template=TEMPLATE,
                              out_dir=OUTDIR,
                              **kwargs)
            uid = ''.join(random.choices(ascii_lowercase, k=5))
            evo.write_json({uid:output}, scores_path)
            print('\033[0;36m written')
            return output['score']
        except Exception as e:
            print('\033[0;36m' + e)
            return 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pop_size',type=int, default=10)
    parser.add_argument('-n','--n_generations',type=int, default=5)
    parser.add_argument('-e','--exhaustiveness',type=int, default=16)
    parser.add_argument('-s','--survival',type=float, default=0.25)
    parser.add_argument('-o','--outdir', default='runs')
    args = parser.parse_args()
    main(args)
