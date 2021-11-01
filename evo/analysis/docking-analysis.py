import os
import sys
import json
from multiprocessing.pool import ThreadPool

import pandas as pd
import prody as pr

pr.confProDy(verbosity='warning')

RESNAMES = {'ALA':'A',
            'ARG':'R',
            'ASN':'N',
            'ASP':'D',
            'CYS':'C',
            'GLN':'G',
            'GLU':'E',
            'GLY':'G',
            'HEM':'X',
            'HIS':'H',
            'ILE':'I',
            'LEU':'L',
            'LYS':'K',
            'MET':'M',
            'PHE':'F',
            'PRO':'P',
            'SER':'S',
            'THR':'T',
            'TRP':'W',
            'TYR':'Y',
            'VAL':'V'}

def get_mutants(root):
    # return path list
    return [i.replace('\n','')  for i in os.popen(f'find {root} -mindepth 2 -type d ')]

def analysis(mutant):
    # combine analysis functions
    d1 = contacts(mutant)
    d2 = binding_energies(mutant)
    for i,j in zip(d1,d2):
        if i in d2:
            d1[i]['aff'] = d2[i]
        else:
            print(mutant, i)
    return {'Mutant':mutant,
            'Sequence':get_sequence(mutant),
            'Docking':d1}

def contacts(mutant, radius=4.0):
    # return ligands:contact_res
    target = pr.parsePDB(os.path.join(mutant,'clean_receptor.pdb'))
    target_contacts_obj = pr.Contacts(target)
    ligands = {i.split('.')[0]:pr.parsePDB(os.path.join(mutant, i)) for i in os.listdir(mutant) if 'mode' in i}
    contacts = {i:target_contacts_obj(radius, ligands[i]) for i in ligands}
    def get_res_from_contacts(c):
        return dict(zip([str(i) for i in c.getResnums()],
                        [RESNAMES[i] for i in c.getResnames()]))
        #return {'resnums':[str(i) for i in c.getResnums()], 
        #        'resnames':[RESNAMES[i] for i in c.getResnames()]}
    return {i:get_res_from_contacts(contacts[i]) for i in contacts}

def binding_energies(mutant):
    # return ligand:energy
    df = pd.read_csv(os.path.join(mutant, 'scores.csv'))[['mode','affinity (kcal/mol)']]
    ##########3 this crashed the programt last time
    try:
        return {f'mode{int(i)}':j for i, j in zip(df['mode'], df['affinity (kcal/mol)'])}
    except:
        raise Warning
        return {}

def get_sequence(mutant):
    target = pr.parsePDB(os.path.join(mutant,'clean_receptor.pdb'))
    return dict(zip([str(i) for i in target.getResindices()], 
                    [RESNAMES[i] for i in target.getResnames()]))


def main(root):
    mutant_dirs = get_mutants(root)

    #with ThreadPool() as process_pool:
    #    results = process_pool.map(analysis, mutant_dirs)

    from tqdm import tqdm
    results = []
    try: # write file in case of interrupt
        for i in tqdm(mutant_dirs):
            results.append(analysis(i))
    except:
        pass

    with open('docking-analysis.json','a') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main(sys.argv[1])
