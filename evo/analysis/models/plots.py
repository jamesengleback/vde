import torch
import matplotlib.pyplot as plt
import seaborn as sns

def scatter(x,y, title):
    plt.figure(figsize=(10,10))
    plt.scatter(x=x, y=y, s=0.5)
    plt.plot([0,1],[0,1], color='yellow')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Model Test Set Predictions' + title)
    plt.savefig(f'{title}.png')
    plt.close()

def kde(x,y, title):
    plt.figure(figsize=(10,10))
    sns.kdeplot(x=x.reshape(-1), y=y.reshape(-1))
    plt.plot([0,1],[0,1])
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Model Test Set Predictions' + title)
    plt.savefig(f'{title}.png')
    plt.close()

def loss_plot(loss_record, sigma=5):
    from scipy.ndimage import gaussian_filter1d 
    smoothed = gaussian_filter1d(loss_record.numpy(), sigma)
    plt.figure(figsize=(15,10))
    plt.plot(loss_record, alpha=0.5)
    plt.plot(smoothed, alpha=0.5)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('Loss-record.png')
    plt.close()


def main():
    plt.style.use('dark_background')

    loss_record = torch.load('loss_record.pt')
    loss_plot(loss_record)
    # ---

    c20_act = torch.load('y1_hs.pt')
    c20_pred = torch.load('y1_is.pt')

    scatter(c20_pred, c20_act, 'c20-scatter')
    kde(c20_pred, c20_act, 'c20-kde')

    # ---

    aff_act = torch.load('y2_hs.pt')
    aff_pred = torch.load('y2_is.pt')
    
    scatter(aff_pred, aff_act, 'aff-scatter')
    kde(aff_pred, aff_act, 'aff-kde')

if __name__ == '__main__':
    main()
