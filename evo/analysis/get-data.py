import os
import pandas as pd

def main():
    files = [i.replace('\n','') for i in os.popen('find ../runs -maxdepth 2 -name scores.csv')]
    df = pd.concat([pd.read_csv(i) for i in files], axis=0)
    df.to_csv('all_scores.csv')


if __name__ == '__main__':
    main()
