import enz
import numpy as np

def c20_fe_distance(protein, vina_pose):
    fe = protein.df.loc[protein.df['element_symbol'] == 'FE',\
            ['x_coord', 'y_coord', 'z_coord']].values
    c20 = vina_pose.df.loc[vina_pose.df['atom_number'] == 20,\
            ['x_coord', 'y_coord', 'z_coord']].values
    return np.linalg.norm(fe - c20)

def mean_dists_affs(protein, results):
    '''
    returns mean distance, mean affinity
    '''
    affinities = np.array([i['affinity'] for i in results]).astype(float)
    distances = np.array([c20_fe_distance(protein, i['mol']) for i in results])
    return distances.mean(), affinities.mean()

def score_a(protein, results):
    #### dodgy
    affinities = np.array([i['affinity'] for i in results]).astype(float)
    distances = np.array([c20_fe_distance(protein, i['mol']) for i in results])
    return np.mean(distances) - np.log(abs(affinities).mean())

def score_b(protein, results):
    affinities = np.array([i['affinity'] for i in results]).astype(float)
    distances = np.array([c20_fe_distance(protein, i['mol']) for i in results])
    def softmax(arr):
        e = np.exp(arr - max(arr))
        return e / sum(e)
    return np.mean(distances * softmax(abs(affinities)))

def test():
    p = enz.protein('DM-Mesotrione/clean_receptor.pdb',
            cofactors = ['HEM'],
            key_sites = [49,75,82, 87, 181, 188, 263, 330, 400])

    mesotrione = 'CS(=O)(=O)C1=CC(=C(C=C1)C(=O)C2C(=O)CCCC2=O)[N+](=O)[O-]'

    r = p.dock(mesotrione,
            exhaustiveness = 2)
    
    print(score_mesotrione(p, r))

if __name__ == '__main__':
    test()
