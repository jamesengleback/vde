import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
sys.path.append('../..')
from bm3 import MXN_SITES

MXN_SITES = [i-3 for i in MXN_SITES]

def detect_delim(path):
    # detect file type
    import re
    with open(path, 'r') as f:
        head = f.read(1000).split('\n')[:-1]
    delimiters = [',','\t',';']
    n_occurences = lambda s, q : len(re.findall(q,s)) 
    for i in delimiters:
        if i in ''.join(head):
            if len(set([n_occurences(s, i) for s in head])) == 1:
                return i


def read_file(path):
    delim = detect_delim(path)
    return pd.read_csv(path, delimiter=delim).drop_duplicates()

def get_genes(sequences):
    return pd.DataFrame([[i[j] for j in MXN_SITES] for i in sequences],
                        columns=MXN_SITES)

def get_counts(genes_df):
    return pd.DataFrame([genes_df[i].value_counts() for i in genes_df],
                         columns=list('RHKDESTNQCGPAVILMFYW')).fillna(0).T

def heatmap(counts, output):
    aa_freq = counts / counts.sum(axis=0)
    plt.figure(figsize=(15,10))
    sns.heatmap(aa_freq, 
            square=False, 
            cbar_kws={'orientation':'horizontal',
                      'label':'Relative Frequency'},
            linewidths=0.01, 
            linecolor='0.1'
            )
    plt.title('Observerd Relative Amino Acid Frequency')
    plt.savefig(output, transparent=True)
    plt.close()



def logo(counts, output):
    cols = counts.columns
    aa_freq = counts.copy() / counts.sum(axis=0)
    aa_freq.columns= range(len(aa_freq.columns))
    logo = logomaker.Logo(aa_freq.T, color_scheme='chemistry', figsize=([15,5]))
    logo.ax.set_xticks(range(len(cols)), minor=False)
    logo.ax.set_xticklabels(cols, rotation = 90)
    plt.ylabel('Frequency')
    plt.xlabel('Position')
    plt.title('Observerd Amino Acid Frequency')
    plt.savefig(output, transparent=True)
    plt.close()

def main(args):
    INPUT = args.input
    OUTPUT = args.output
    if args.dark:
        plt.style.use('dark_background')

    df = read_file(INPUT)
    genes = get_genes(df['sequence'])
    counts = get_counts(genes)

    if args.heatmap:
        heatmap(counts, OUTPUT)

    if args.logo:
        logo(counts, OUTPUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input')
    parser.add_argument('-o','--output')
    parser.add_argument('-d','--dark', action='store_true')
    parser.add_argument('-hm','--heatmap', action='store_true')
    parser.add_argument('-l','--logo', action='store_true')
    args = parser.parse_args()
    main(args)
