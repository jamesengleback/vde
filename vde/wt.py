import os
import argparse
import pandas as pd
from utils import BM3_DM, DOCKING_SITE, MXN_SITES
from sfxns import score_a, score_b, score_c
from main import evaluate

BLUE='\033[0;36m '


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    wt_gene = ''.join([BM3_DM[i] for i in MXN_SITES])
    score = evaluate(gene=wt_gene,
                     out_dir_root=args.outdir,
                     score_fn=score_c,
                     template=wt_gene,
                     exhaustiveness=args.exhaustiveness)
    pd.Series(score).to_csv(os.path.join(args.outdir, 'score.csv'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--exhaustiveness',type=int, default=1)
    parser.add_argument('-o','--outdir', default='wt')
    parser.add_argument('-fn','--score_fn', default='a')
    args = parser.parse_args()
    main(args)
