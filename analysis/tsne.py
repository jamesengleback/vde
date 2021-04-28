import sys
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def featurize_gene(gene):
    return pd.DataFrame([AA_FEATURES[i] for i in gene])

def featurize_aa(aa):
    mol = Chem.MolFromFASTA(aa)
    mol = Chem.AddHs(mol)
    return {'FormalCharge' : Chem.GetFormalCharge(mol),
            'NumAliphaticCarbocycles' : Chem.CalcNumAliphaticCarbocycles(mol),
            'NumAliphaticHeterocycles' : Chem.CalcNumAliphaticHeterocycles(mol),
            'NumAliphaticRings' : Chem.CalcNumAliphaticRings(mol),
            'NumAmideBonds' : Chem.CalcNumAmideBonds(mol),
            'NumAromaticCarbocycles' : Chem.CalcNumAromaticCarbocycles(mol),
            'NumAromaticHeterocycles' : Chem.CalcNumAromaticHeterocycles(mol),
            'NumAromaticRings' : Chem.CalcNumAromaticRings(mol),
            'NumBridgeheadAtoms' : Chem.CalcNumBridgeheadAtoms(mol),
            'NumHBA' : Chem.CalcNumHBA(mol),
            'NumHBD' : Chem.CalcNumHBD(mol),
            'NumHeteroatoms' : Chem.CalcNumHeteroatoms(mol),
            'NumHeterocycles' : Chem.CalcNumHeterocycles(mol),
            'NumLipinskiHBA' : Chem.CalcNumLipinskiHBA(mol),
            'NumLipinskiHBD' : Chem.CalcNumLipinskiHBD(mol),
            'NumRings' : Chem.CalcNumRings(mol),
            'NumRotatableBonds' : Chem.CalcNumRotatableBonds(mol),
            'NumSaturatedCarbocycles' : Chem.CalcNumSaturatedCarbocycles(mol),
            'NumSaturatedHeterocycles' : Chem.CalcNumSaturatedHeterocycles(mol),
            'ExactMolWt' : Chem.CalcExactMolWt(mol),
            'NumSaturatedRings' : Chem.CalcNumSaturatedRings(mol),
            'FractionCSP3' : Chem.CalcFractionCSP3(mol),
            'NumSpiroAtoms' : Chem.CalcNumSpiroAtoms(mol)}


AA_FEATURES = {i:featurize_aa(i) for i in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']}

def plot(coords, scores=None, savepath = None):
    plt.figure(figsize = (10,10))
    plt.set_cmap('inferno')
    if scores is None:
        plt.scatter(coords[:,0], coords[:,1], s= 10)
    else:
        plt.scatter(coords[:,0], coords[:,1], c = scores, s = 10)
        plt.colorbar(label = 'Fitness', font = 14)
    plt.axis('off')
    plt.title('t-SNE of Mutants mapped to Fitness', font = 14)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def plot_label(coords, labels, savepath = None):
    assert len(coords) == len(labels)
    if coords is not pd.DataFrame:
        coords = pd.DataFrame(coords)
    coords['label'] = labels
    plt.figure(figsize = (10,10))
    legend_handles =[]
    for i in sorted(list(set(labels))):
        legend_handles.append(f'cluster: {i}')
        cluster = coords.loc[coords['label'] == i,:]
        plt.scatter(cluster.loc[:,0], cluster.loc[:,1], s = 10)
    legend_handles[0] = 'no cluster'
    plt.axis('off')
    plt.legend(legend_handles, title = 'cluster id', font = 14)
    plt.title('t-SNE of Mutants Clustered with DBSCAN', font = 14)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

def main(path):
    df = pd.read_csv(path, index_col = 0)
    genes = df['gene']
    scores = df['score'] 
    scores -= scores.min()
    scores /= scores.max()
    scores = 1 - scores


    x = np.vstack([featurize_gene(i).values.astype(float).reshape(-1) for i in genes])

    tsne = TSNE(perplexity = 40)
    coords = tsne.fit_transform(x)
    plot(coords, scores, 'moclv-ft-tsne.png')
    coords = pd.DataFrame(coords, index = genes)

    dbscan = DBSCAN(eps = 5)
    labels = dbscan.fit_predict(coords)
    plot_label(coords, labels, 'moclv-ft-tsne-clusters.png')
    coords['label'] = labels
    coords.to_csv('coords.csv')

if __name__ == '__main__':
    #main(sys.argv[1])
    main('../runs/moclv/scores.csv')
