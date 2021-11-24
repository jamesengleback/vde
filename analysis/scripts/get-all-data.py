import sys
import os
import os.path as osp
import re
import json
import pandas as pd
from multiprocessing.pool import ThreadPool


RESNAMES = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D',
            'CYS':'C', 'GLN':'G', 'GLU':'E', 'GLY':'G',
            'HEM':'X', 'HIS':'H', 'ILE':'I', 'LEU':'L',
            'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
            'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y',
            'VAL':'V'}

class PDB:
    def __init__(self,
                 path):
        self.pdb_path = path # avoids name clash in inherited
    @property
    def structure(self):
        import prody as pr
        return pr.parsePDB(self.pdb_path)
    @property
    def df(self):
        get_info = lambda atom : {'index':atom.getIndex(),
                                  'element':atom.getData('element'),
                                  'name':atom.getData('name'),
                                  'beta':atom.getData('beta'),
                                  'chain':atom.getData('chain'),
                                  'x':(coords := atom.getCoords())[0],
                                  'y':coords[1],
                                  'z':coords[2],
                                  }
        return pd.DataFrame([get_info(atom) for atom in self.structure])

class Structure(PDB):
    def __init__(self,
                 path):
        assert osp.isfile(path)
        super().__init__(path)
    @property
    def sequence(self):
        d = dict(zip(self.structure.getResindices(), # data is by atom so this strips dups
                     [RESNAMES[i] for i in self.structure.getResnames()] ))
        return ''.join([d[i] if i in d else '-'  for i in range(max(d.keys()))])

        

class Pose(PDB):
    def __init__(self,
                 path):
        super().__init__(path)
    @property
    def name(self):
        return f"{osp.basename(self.path).replace('.pdb','')}"
    @property
    def receptor_path(self):
        return osp.join(osp.dirname(self.path), 'clean_receptor.pdb')
    @property
    def receptor_structure(self):
        return Structure(self.receptor_path)
    @property
    def contacts(self):
        pass
    @property
    def affinity(self):
        df = pd.read_csv(osp.join(osp.dirname(self.path), 'scores.csv'), index_col=0)
        d = dict(zip([f'mode{int(i)}' for i in df['mode']],
                     df['affinity (kcal/mol)']))
        return d[self.name]
    def __repr__(self):
        return f"pose: {self.path}"

def c20_dist(structure, pose):
    pass

def default_heme_fe_coords(path='../../../data/4KEY.pdb'):
    s = Structure(path)
    fe = s.df.loc[s.df['element'] == 'FE',:]
    return fe.loc[fe['chain'] == 'A',['x','y','z']]

class Mutant(Structure):
    def __init__(self,
                 path):
        assert osp.exists(path)
        assert osp.isdir(path)
        super().__init__(osp.join(path, 'clean_receptor.pdb'))
        self.path = path
    @property
    def config(self):
        return read_config(osp.join('..','config.json'))
    @property
    def scores(self):
        df = pd.read_csv(osp.join(self.path, 'scores.csv'), index_col=0)
        df.columns = ['mode',  'affinity', 'dist_from_best_ub', 'dist_from_best_lb',]
        df['mode'] = df['mode'].apply(lambda n : f'mode{int(n)}')
        return df
    @property
    def poses(self):
        pose_paths = filter(lambda path : 'mode' in path, os.listdir(self.path))
        pose_paths_full = map(lambda path : osp.join(self.path, path), pose_paths)
        for i in pose_paths_full:
            yield Pose(i)
    @property
    def affinities(self):
        return dict(zip(self.scores['mode'], self.scores['affinity']))
    @property
    def mean_affinity(self):
        return sum(a := self.affinities.values()) / len(a)
    @property
    def contacts(self):
        pass # return [pose.contacts for pose in self.poses]
    @property
    def c20_dists(self):
        import numpy as np
        from numpy.linalg import norm
        dist = lambda a, b : norm(np.array(a) - np.array(b))
        if 'FE' in self.df['element']:
            fe_xyx = (struc := self.df).loc[struc['element'] == 'FE',list('xyz')]
        else:
            fe_xyx = default_heme_fe_coords()
        c20_xyz = lambda pose: pose.df.loc[20,list('xyz')]
        return [dist(fe_xyx, c20_xyz(i)) for i in self.poses]
    @property
    def mean_c20_dists(self):
        return  sum(self.c20_dists) / len(self.c20_dists)
    @property
    def contains_hem(self):
        return 'X' in self.sequence
    def __repr__(self):
        return f"mutant: {self.path}"



class Run:
    def __init__(self, 
                 path,
                 *args):
        assert os.path.exists(path)
        self.path = path
        self.mutant_dirs = lambda : filter(lambda : True if 'clean_receptor.pdb' in os.listdir(self.path) else False, args)
    def __len__(self):
        return len(list(self.mutant_dirs()))
    def __repr__(self):
        return f'{self.path}'
    def __iter__(self):
        for i in self.mutants:
            yield i
    @property
    def mutants(self):
        childs = map(lambda path : osp.join(self.path, path), os.listdir(self.path))
        child_dirs = filter(lambda path : osp.isdir(path), childs)
        mutant_dirs = filter(lambda path : True if 'clean_receptor.pdb' in os.listdir(path) else False, child_dirs)
        return map(Mutant, mutant_dirs) 

    @property
    def config(self):
        return read_config(osp.join(self.path, 'config.json'))
    @property
    def scores(self):
        if osp.exists(a := osp.join(self.path, 'scores.csv')):
                return pd.read_csv(a)



def read_config(path):
    if osp.exists(path):
        with open(path,'r') as f:
            return json.loads(f.read())

def get_runs(root):
    # ---
    child_dirs = lambda path : filter(lambda path : True if i in path else False, find)
    contains_config = lambda path : True if 'config.json' in path else False
    # ---
    find = os.popen(f'find {root}').read().splitlines()
    run_dirs = map(lambda path : path.replace('config.json',''), 
                   filter(contains_config, find))
    # ---
    return map(Run, run_dirs)

def get_mutants(path, _l=[]):
    is_mutant_path = lambda path : osp.exists(osp.join(path, 'clean_receptor.pdb'))
    ls = lambda path : [osp.join(path,i) for i in os.listdir(path)]
    # ---
    if osp.isdir(path):
        for i in ls(path):
            if is_mutant_path(path):
                _l.append(Mutant(path))
            else:
                get_mutants(i, _l)
        return _l


def main(roots):
    cols = {
            'path': lambda i : i.path,
            'mean_affinity': lambda i : i.mean_affinity,
            'config': lambda i : i.config,
            'c20_dists' : lambda i : i.c20_dists,
            'affinities' : lambda i : i.affinities,
            'mean_affinity' : lambda i : i.mean_affinity,
            'c20_dist_mean' : lambda i : i.mean_c20_dists,
            'sequence': lambda i : i.sequence,
            'contains_hem':lambda i : i.contains_hem,
            }
    fmt = lambda l : '\t'.join(map(str, l)) + '\n'
    sys.stdout.write(fmt(cols.keys()))
    def helper(mutant_path):
        sys.stdout.write(fmt([fn(mutant_path) for fn in cols.values()]))
    for root in roots:
        mutants = get_mutants(root)
        with ThreadPool(30) as pool:
            results = pool.map(helper, mutants)

if __name__ == '__main__':
    main(sys.argv[1:])
