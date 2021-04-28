import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generation_num(idx):
    # ffs
    l = []
    counter = 0
    last = 0
    for i in idx:
        if i < last:
            counter += 1
        l.append(counter)
        last = i
    return l

def main():
    df = pd.read_csv('../runs/moclv/scores.csv')
    df['fitness'] = 1 - ((df['score'] - df['score'].min()) / df['score'].max())
    df['gen'] = pd.Series(generation_num(df['Unnamed: 0']))

    plt.figure(figsize=(15,4))
    sns.violinplot(x=df['gen'], y=df['fitness'])
    plt.ylabel('Fitness')
    plt.xlabel('Generation Number')
    plt.savefig('molcv-fitness-violin.png')
    plt.show()


if __name__ == '__main__':
    main()
