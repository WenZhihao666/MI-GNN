import os.path as osp
import time
from torch_geometric.datasets import Flickr, Yelp
from torch_geometric.utils import k_hop_subgraph
import os
import torch
from torch_geometric.data import Data

def sub_data_maker(data_name):

    name = 'Sub_{}'.format(data_name)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', data_name)
    if data_name == "Flickr":
        dataset = Flickr(path, data_name)
    else:
        dataset = Yelp(path, data_name)

    print(data_name)
    start = time.perf_counter()
    f_data = []
    for i in range(0, 80000, 100):
        adj = k_hop_subgraph(i, 1, dataset.data.edge_index, relabel_nodes=True)[1]
        index = k_hop_subgraph(i, 1, dataset.data.edge_index, relabel_nodes=True)[0].numpy()
        feature = dataset.data.x[index]
        label = dataset.data.y[index]
        data = Data(x=feature, edge_index=adj, y=label)
        f_data.append(data)

    os.makedirs('./data/{}/processed'.format(name), exist_ok=True)
    torch.save(f_data, './data/{}/processed/data.pt'.format(name))

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))

    print(f_data[:10])

if __name__ == '__main__':

    sub_data_maker('Yelp') #if want Flickr, change 'Yelp' into 'Flickr'
