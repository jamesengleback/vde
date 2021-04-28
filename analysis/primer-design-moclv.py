import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import SimDivFilters
import mxn
import sys
sys.path.append('../evo')
from bm3 import BM3_DM, ACTIVE_SITE_AAS

pd.options.display.width=200

WT = 'TYLFVLLIA'
orf = 'ATGGGCAGCAGCCATCATCATCATCATCACAGCAGCGGCCTGGTGCCGCGCGGCAGCcatatgacaattaaagaaatgcctcagccaaaaacgtttggagagcttaaaaatttaccgttattaaacacagataaaccggttcaagctttgatgaaaattgcggatgaattaggagaaatctttaaattcgaggcgcctggccgtgtaacgcgctacttatcaagtcagcgtctaattaaagaagcatgcgatgaatcacgctttgataaaaacttaagtcaagcgcttaaatttgtacgtgattttgcaggagacgggttatttacaagctggacgcatgaaaaaaattggaaaaaagcgcataatatcttacttccaagcttcagtcagcaggcaatgaaaggctatcatgcgatgatggtcgatatcgccgtgcagcttgttcaaaagtgggagcgtctaaatgcagatgagcatattgaggtaccggaagacatgacacgtttaacgcttgatacaattggtctttgcggctttaactatcgctttaacagcttttaccgagatcagcctcatccatttattacaagtatggtccgtgcactggatgaagcaatgaacaagctgcagcgagcaaatccagacgacccagcttatgatgaaaacaagcgccagtttcaagaagatatcaaggtgatgaacgacctagtagataaaattattgcagatcgcaaagcaagcggtgaacaaagcgatgatttattaacgcacatgctaaacggaaaagatccagaaacgggtgagccgcttgatgacgagaacattcgctatcaaattattacattcttaattgcgggacacgaaacaactagtggtcttttatcatttgcgctgtatttcttagtgaaaaatccacatgtattacaaaaagcagcagaagaagcagcacgagttctagtagatcctgttccaagctacaaacaagtcaaacagcttaaatatgtcggcatggtcttaaacgaagcgctgcgcttatggccaactgctcctgcgttttccctatatgcaaaagaagatacggtgcttggaggagaatatcctttagaaaaaggcgacgaactaatggttctgattcctcagcttcaccgtgataaaacaatttggggagacgatgtggaagagttccgtccagagcgttttgaaaatccaagtgcgattccgcagcatgcgtttaaaccgtttggaaacggtcagcgtgcgtgtatcggtcagcagttcgctcttcatgaagcaacgctggtcctaggtatgatgctaaaacactttgactttgaagatcatacaaactacgagctggatattaaagaaactttaacgttaaaacctgaaggctttgtggtaaaagcaaaatcgaaaaaaattccgcttggcggtattccttcacctagcactgaacagtctgctaaaaaagtacgcaaatag'

def hamming(a,b):
    return sum([1 for i,j in zip(a,b) if i != j])

def hamming_violin_plot(df):
    plt.figure(figsize=(15,4))
    sns.violinplot(x=df['wt-dist'], y=df['fitness'])
    plt.ylabel('Fitness')
    plt.xlabel('Hamming Distance From Wild Type')
    plt.title('Fitness of BM3 Mutants Against Sequence Distance From Wild Type')
    plt.savefig('hamming-violin.png')
    #plt.show()
    plt.close()

def hammingG(genes):
    genes = [WT] + genes.to_list()
    edge_weights = np.array([[hamming(i,j)  for i in genes] for j in genes])
    G = nx.from_numpy_matrix(edge_weights)
    for i,j in zip(G.nodes, genes):
        G.nodes[i]['gene']=j
    return G

def get_mutation(a, b): # gene strings
    return {i:k for i, j, k in zip(ACTIVE_SITE_AAS, a,b) if j != k}

def maxmin(genes, n, weights=None):
    fn = lambda i, j : hamming(genes[i], genes[j])
    picker = SimDivFilters.MaxMinPicker()
    idx = picker.LazyPick(fn, len(genes), n)
    return idx

def main():
    df = pd.read_csv('../runs/moclv/scores.csv')
    df['fitness'] = 1 - ((df['score'] - df['score'].min()) /df['score'].max()) 
    #df['wt-dist'] = [hamming(i, WT) for i in df['gene']]
    best = df.loc[df['fitness'] > 0.7,:]
    best_genes = best['gene'].reset_index(drop=True)
    idx = list(maxmin(best_genes.to_list(),8))
    selection = best_genes[idx]

    mutations = pd.DataFrame([get_mutation(WT, i) for i in selection])
    fitnesses = [float(df.loc[df['gene'] ==i,'fitness']) for i in selection]
    mutations['fitness'] = fitnesses
    mutations.to_latex('mutations.tex')
    mutations = mutations.drop('fitness',axis=1).melt().drop_duplicates().dropna()
    mutations = [[i,j] for i,j in zip(mutations['variable'], mutations['value'])]
    cds = mxn.CDS(orf, BM3_DM)
    for i in mutations:
        cds.mutate(i[0], i[1])
    pd.DataFrame(cds.primers).T.to_latex('primers.tex')

if __name__ == '__main__':
    main()
