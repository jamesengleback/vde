import fire
import data
import model
import train
import utils

def test_data(encoding='ohe'):
    assert encoding in {'ohe', 'chem'}
    if encoding == 'ohe':
        dataset = data.Data(path='../outputs/newscore2-98k.csv',
                         template_path='../../../data/4KEY.pdb',
                         encoding='ohe',
                         N_DIVISIONS=16,
                         test=True)
    elif encoding == 'chem':
        dataset = data.Data(path='../outputs/newscore2-98k.csv',
                         template_path='../../../data/4KEY.pdb',
                         encoding='ohe',
                         N_DIVISIONS=16,
                         test=True)

def test_model(dim=16, downsamples=4, recycles=4):
    dataset = data.Data(path='../outputs/newscore2-98k.csv',
                     template_path='../../../data/4KEY.pdb',
                     N_DIVISIONS=16,
                     test=True)
    v_0, a_0, d_0, s_0 = dataset[0] # voxels, affinity, distance, score
    net = model.Model(*v_0.shape, 
                      dim=dim,
                      n_downsamples=downsamples,
                      n_recycles=recycles,
                      ) # init from shape
    yh = net(v_0.unsqueeze(0))

def test_train():
    pass

def test_utils(encoding='ohe'):
    dataset = data.Data(path='../outputs/newscore2-98k.csv',
                     template_path='../../../data/4KEY.pdb',
                     N_DIVISIONS=16,
                     encoding=encoding,
                     test=True)
    data_0 = dataset[0]
    for v,a,b,c in dataset:
        print(v.shape)

if __name__ == '__main__':
    fire.Fire()
