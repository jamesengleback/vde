#!/usr/bin/env python
import sys
import os
import re
import pandas as pd

find = lambda arg : os.popen(f"find {arg}").read().split('\n')
grep = lambda pat, ls : list(filter(lambda i : re.search(pat, i) is not None, ls))
grepv = lambda pat, ls : list(filter(lambda i : re.search(pat, i) is None, ls))

def main(arg):
    walk = find(arg)
    run_metadata = list(filter(lambda i : os.path.isfile(i), grepv("[A-Z]{25}", walk)))
    csvs = grep('csv', run_metadata)
    
    df = pd.concat([pd.read_csv(i) for i in csvs]).drop_duplicates()
    df.columns = ['gene','aff_mean','dist_mean', 'score', 'ham', 'uid']
    df.to_csv(sys.stdout, index=False)

if __name__ == '__main__':
    main(sys.argv[1])
