import enz
import numpy as np

BLUE='\033[0;36m '

def softmax(arr):
    e = np.exp(arr - max(arr))
    return e / sum(e)

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
    score = np.mean(distances) - np.log(abs(affinities).mean())
    return {'score': score, 
            'distance_mean':distances.mean(),
            'affinities_mean':affinities.mean()}

def score_b(protein, results):
    affinities = np.array([i['affinity'] for i in results]).astype(float)
    distances = np.array([c20_fe_distance(protein, i['mol']) for i in results])
    score = np.mean(distances * softmax(abs(affinities)))
    return {'score': score, 
            'distance_mean':distances.mean(),
            'affinities_mean':affinities.mean()}

def score_c(protein, results):
    affinities = np.array([i['affinity'] for i in results]).astype(float)
    distances = np.array([c20_fe_distance(protein, i['mol']) for i in results])
    softAff = softmax(abs(affinities))
    e = sum(affinities)
    score = np.mean(softAff * distances)  - np.log(abs(e))
    return {'score': score, 
            'distance_mean':distances.mean(),
            'affinities_mean':affinities.mean()}
