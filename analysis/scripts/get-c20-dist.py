import sys
sys.path.append('..')
import os
import os.path as osp
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
import enz

def get_dist(prot_path, lig_path):
    # hemes are missing, so i'm doing the ligating sulphur of cys400
    prot = enz.protein(prot_path)
    lig = enz.mol(lig_path)
    cys400 = prot.df.loc[prot.df['residue_number']==400,:]
    s = cys400.loc[cys400['element_symbol'] == 'S',['x_coord','y_coord','z_coord']].values
    c20 = lig.df.loc[lig.df['atom_number'] == 20,\
            ['x_coord', 'y_coord', 'z_coord']].values
    return np.linalg.norm(s-c20)

def dir_to_distances(path):
    assert osp.isdir(path)
    modes = [osp.join(path,i) for i in os.listdir(path) if 'mode' in i]
    receptor = osp.join(path, 'clean_receptor.pdb')
    scores = pd.read_csv(osp.join(path,'scores.csv'), index_col=0)
    affinities = {f'mode{int(i)}':j for i, j in zip(scores['mode'], scores['affinity (kcal/mol)'])}
    x = {osp.basename(i).split('.')[0]:get_dist(receptor,i) for i in modes}
    df = pd.DataFrame([x,affinities], index=['c400S-c20-dist','affinity']).T
    df['mutant'] = path.split('/')[-1]
    return df


def main(root):
    dirs = [osp.dirname(i.replace('\n','')) for i in \
            os.popen(f'find {root} -name *pdb')]

    with ThreadPool() as process_pool :
        results = process_pool.map(dir_to_distances, dirs) 
    df = pd.concat(results)
    df.to_csv(sys.stdout)

if __name__ == '__main__':
    main(sys.argv[1])
