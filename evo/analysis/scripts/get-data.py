import sys
import os
import pandas as pd

def main(args):
    files = []
    for root in args:
        files = files + [i.replace('\n','') for i in os.popen(f'find {root} -maxdepth 2 -name scores.csv')]
    df = pd.concat([pd.read_csv(i) for i in files], axis=0)

    df.to_csv(sys.stdout)

if __name__ == '__main__':
    main(sys.argv[1:])
