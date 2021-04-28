import enz
import numpy as np

def c20_fe_distance(protein, vina_pose):
    fe = protein.df.loc[protein.df['element_symbol'] == 'FE',\
            ['x_coord', 'y_coord', 'z_coord']].values
    c20 = vina_pose.df.loc[vina_pose.df['atom_number'] == 20,\
            ['x_coord', 'y_coord', 'z_coord']].values
    return np.linalg.norm(fe - c20)

def score_mesotrione(p, r):
    pose_dict = r.dictionary
    affinities = np.array([pose_dict[i]['affinity'] for i in pose_dict]).astype(float)
    distances = np.array([c20_fe_distance(p, pose_dict[i]['mol']) for i in pose_dict])
    aff_sq = affinities ** 2

    return sum(distances / aff_sq), distances.mean(), affinities.mean()


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
