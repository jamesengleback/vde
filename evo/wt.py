import enz
import os
import numpy as np
from bm3 import BM3_DM, MXN_SITES
from score import score_mesotrione 

savedir = 'wt'

def evaluate(gene):
    p = enz.protein('../data/4KEY.pdb',
                    seq = BM3_DM, # my residue numbering system
                    cofactors = ['HEM']) # keep the heme


    docking_results = p.dock('CCS(=O)(=O)C1=C(N2C=CC=CC2=N1)S(=O)(=O)NC(=O)NC3=NC(=CC(=N3)OC)OC',
                             target_residues = MXN_SITES,
                             exhaustiveness = 16)
    docking_results.save(os.path.join(savedir, gene))
    return score_mesotrione(p, docking_results) 


def main():
    os.makedirs(savedir, exist_ok=True)
    score = evaluate('wt')
    with open(os.path.join(savedir, 'wt-score'), 'w')  as f:
        f.write(str(score))

if __name__ == '__main__':
    main()
