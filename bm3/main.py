import os
import os.path as osp
import random
from string import ascii_lowercase
import json
from tqdm import tqdm
import argparse

import enz
import ga
import evo

from bm3 import BM3_DM, MXN_SITES, DOCKING_SITE
from score import score_mesotrione, mean_dists_affs


def evaluate(gene,
             template=None,
             exhaustiveness=16,
             out_dir=None):
    '''
    mutant evaluation function specific to this project
    '''
    #print(f'\033[0;36m {gene}')
    assert len(gene) == len(MXN_SITES)
    out_dir = osp.join(out_dir,gene) if out_dir is not None else None
    sequence = evo.mutate_string(BM3_DM, dict(zip(MXN_SITES, gene))) # mutates string by dictionary of pos:aa
    protein, docking_results = evo.simulate(structure_template='4KEY.pdb',
                                            sequence=sequence,
                                            keep=['HEM'],
                                            ligand_smiles='CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]',
                                            binding_site=DOCKING_SITE,
                                            out_dir=out_dir,
                                            tmp_suffix='',
                                            exhaustiveness=exhaustiveness)
    #print('\033[0;36m docked')
    dist_mean, aff_mean = mean_dists_affs(protein, docking_results)
    #print('\033[0;36m distances')
    score = sum([dist_mean, aff_mean])
    if template is not None:
        ham = evo.hamming(template, gene)
        score += ham
    else:
        ham=None
    #print('\033[0;36m ham')
    return {'gene':gene, 
            'score':score, 
            'dist_mean':dist_mean, 
            'aff_mean':aff_mean, 
            'ham':ham}


def main(args):
    # --- config stuff ---
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
    # ---

    def helper(gene):
        # crashes still happen sometimes :(
        try:
            output = evaluate(gene, 
                              exhaustiveness=EXHAUSTIVENESS,
                              template=TEMPLATE,
                              out_dir=OUTDIR)
            uid = ''.join(random.choices(ascii_lowercase, k=5))
            evo.write_json({uid:output}, scores_path)
            print('\033[0;36m written')
            return output['score']
        except Exception as e:
            print('\033[0;36m' + e)
            return 100

    pop = [ga.random_mutate(TEMPLATE) for _ in range(POP_SIZE)] # init
    mxn_layers = ga.Sequential(
                               ga.RandomMutate(),
                               ga.CrossOver(),
                               )
    fn = lambda x : ga.hamming(TEMPLATE,x)
    constrained_mxn_layers = ga.Constrained(
                                    fn=fn,
                                    layers=mxn_layers,
                                    thresh=lambda p : max(map(fn,p)) <=4,
                                    )
    #helper = lambda mutant : sum([i == 'A' for i in mutant])

    pipeline = ga.Sequential(
                             ga.Evaluate(helper, max_workers=POP_SIZE),
                             ga.PickBottom(),
                             ga.Clone(POP_SIZE),
                             constrained_mxn_layers,
                            )

    print(len(pop))
    for _ in tqdm(range(N_GENERATIONS)):
        pop = pipeline(pop)
        #evo.gc(RUN_ID)
        print(len(pop))
        print(pop)
        #print(f'\033[0;36m n mutants: {len(pop)}')
        #print(f'\033[0;36m {pipeline.log}')
        #print(pop)
        #print(f'\033[0;36m n mutants: {len(pop)}')
        #print(f'\033[0;36m end of iteration {_}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pop_size',type=int, default=8)
    parser.add_argument('-n','--n_generations',type=int, default=5)
    parser.add_argument('-e','--exhaustiveness',type=int, default=1)
    parser.add_argument('-s','--survival',type=float, default=0.25)
    parser.add_argument('-o','--outdir', default='runs')
    args = parser.parse_args()
    main(args)
