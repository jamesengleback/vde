import random
import pandas as pd
import torch 
import torch.nn as nn 
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('../runs/moclv/scores.csv', index_col=0).reset_index(drop=True)
    x = torch.from_numpy(\
            pd.get_dummies(pd.DataFrame([list(i) for i in df['gene']])).values).float()
    y = torch.from_numpy(df[['score']].values).float()
    y -= y.min()
    y /= y.max()
    
    idx = list(df.index)
    train_idx = random.choices(idx, k = round(0.8 * len(idx)))
    test_idx = [i for i in idx if i not in train_idx]

    train_x, train_y = x[train_idx], y[train_idx]
    test_x, test_y = x[test_idx], y[test_idx]

    model = nn.Sequential(nn.Linear(x.shape[1], 32, bias=False),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(32, 2, bias=False),
                                    nn.Sigmoid(),
                                    nn.Dropout(),
                                    nn.Linear(2, 1, bias=False),
                                    nn.Dropout(),
                                    nn.Sigmoid())

    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()

    loss_record = []
    for i in tqdm(range(1000)):
        yh = model(train_x)
        loss = loss_fn(yh, train_y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_record.append(loss.detach().item())


    model.eval()
    yh = model(test_x)
    loss = loss_fn(yh, test_y)

    plt.plot(loss_record)
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.show()

    plt.scatter(test_y.detach(), yh.detach())
    plt.plot([0,1],[0,1])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('acutal')
    plt.ylabel('predicted')
    plt.show()

if __name__ == '__main__':
    main()
