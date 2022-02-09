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
    return np.mean(distances * softmax(abs(affinities))), \
           distances.mean(), \
           affinities.mean()

def score_c(protein, results):
    affinities = np.array([i['affinity'] for i in results]).astype(float)
    distances = np.array([c20_fe_distance(protein, i['mol']) for i in results])
    assert len(distances) == len(affinities)
    distexp = np.exp(distances)
    def softmax(arr):
        e = np.exp(arr - max(arr))
        return e / sum(e)
    return sum((distexp * softmax(affinities)) / sum(affinities)) 

BM3_WT='MTIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFAGDGLFTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKVRK*'
BM3_DM='MTIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFVGDGLVTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACIGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKVRK*'
DOCKING_SITE=[49, 51, 75, 78, 82, 87, 88, 184, 188, 226, 252, 255, 260, 263, 290, 295, 328, 330]
MXN_SITES = [47, 49, 51, 75, 78, 88, 94, 138, 142, 175, 178, 184, 188, 205, 226, 252, 255, 260, 263, 290, 295, 328, 330, 350, 353]

def write_json(dictionary, path, mode='a'):
    with open(path,mode) as f:
        json.dump(dictionary,f)

def write_csv(dictionary, path):
    df = pd.DataFrame([dictionary])
    if osp.exists(path):
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, index=False)

def mutate_string(template, target_dict):
    s_ = list(template)
    for i,j in zip(target_dict.keys(), target_dict.values()):
        s_[i] = j
    return ''.join(s_)

def gc(string_match):
    # garbage collection
    # can clash with other enz runs!
    files = [os.path.join('/tmp',i) for i in os.listdir('/tmp')]
    enz_files = [i for i in files if f'{string_match}_enz' in i]
    for i in enz_files:
        if os.path.isfile(i):
            os.remove(i)
        elif os.path.isdir(i):
            shutil.rmtree(i)

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
