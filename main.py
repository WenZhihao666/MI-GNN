from mydataset import MyDataset
import os.path as osp
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn import preprocessing
import numpy as np
import argparse
import torch
import scipy.sparse as sp
from random import sample
import random
from model import Meta_MSNA, Meta_NA
from earlystopping import EarlyStopping
from task_data_generator import task_data_generator


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(
        d_mat_inv_sqrt).tocoo()  


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sgc_precompute(features, adj, degree): 
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_labels_list = []
    graph_adj_list = []
    graph_features_list = []

    if args.dataset == 'DHFR':
        num_attri = 3
        label_dim = 9
        Lab = [1, 2, 3, 4, 5, 6, 7, 8]
    elif args.dataset == 'COX2':
        num_attri = 3
        label_dim = 8
        Lab = [0, 2, 3, 4, 5, 6, 7]
    elif args.dataset == 'Cuneiform':
        num_attri = 3
        label_dim = 7
        Lab = [0, 1, 2, 3, 4, 5]
    elif args.dataset == 'Sub_Flickr':
        num_attri = 500
        label_dim = 7
        Lab = [0, 1, 2, 3, 4, 5]
    elif args.dataset == 'Sub_Yelp':
        num_attri = 300
        label_dim = 10
        Lab = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    social_datasets = {'Sub_Flickr', 'Sub_Yelp'}
    if args.dataset not in social_datasets:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
        datasets = MyDataset(path, args.dataset)
        print('{}'.format(args.dataset))
        loader = DataLoader(datasets)
        for data in loader:
            labels = data.x[:, num_attri:]
            graph_labels_list.append(labels)
            adj = data.edge_index
            graph_adj_list.append(adj)
            features = data.x[:, 0:num_attri]
            graph_features_list.append(features)

    else:
        datasets = torch.load('./data/{}/processed/data.pt'.format(args.dataset))
        i = 0
        if args.dataset == "Sub_Yelp":
            top_10_index = [1, 2, 24, 22, 82, 28, 9, 8, 10, 14]
            top_10_index = np.array(top_10_index, dtype=int)
            top_10_index = torch.from_numpy(top_10_index)
            for a in range(len(datasets)):
                labels = (datasets[a].y).T
                labels = labels[top_10_index]
                labels = labels.T
                graph_labels_list.append(labels)
                adj = datasets[a].edge_index
                graph_adj_list.append(adj)
                features = datasets[a].x
                graph_features_list.append(features)

        else:
            for a in range(len(datasets)):
                labels = datasets[a].y
                graph_labels_list.append(labels)
                adj = datasets[a].edge_index
                graph_adj_list.append(adj)
                features = datasets[a].x
                graph_features_list.append(features)

    train_graphs_index = sample(range(len(graph_labels_list)), int(0.6 * len(graph_labels_list)))
    val_test_graphs_index = list(set(range(len(graph_labels_list))) - set(train_graphs_index))
    test_graphs_index = sample(val_test_graphs_index, int(0.5 * len(val_test_graphs_index)))
    val_graphs_index = list(set(val_test_graphs_index) - set(test_graphs_index))

    def feature_label_generator(index, shuffle=False):
        if shuffle:
            random.shuffle(index)
        reformed_futures_list = []
        reformed_labels_list = []
        neighs_list = []
        adjs_list = []
        for i in index:
            if (args.dataset != 'Cuneiform') & (args.dataset not in social_datasets):
                labels = torch.LongTensor(graph_labels_list[i].long())
                labels = torch.max(labels, dim=1)[1]
            elif args.dataset in ['Cuneiform', 'Sub_Yelp']:
                labels = graph_labels_list[i]
            else:
                labels = torch.LongTensor(graph_labels_list[i].long())
            reformed_labels_list.append(labels.to(device))
            adjs_list.append(graph_adj_list[i].to(device))
            adj = to_scipy_sparse_matrix(graph_adj_list[i],
                                         num_nodes=len(labels))  
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = aug_normalized_adjacency(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj).float()
            adj_0 = adj.to_dense()
            neighs = []
            for a in range(len(labels)):
                neighs.append([])
                for b in range(len(labels)):
                    if adj_0[a][b] != 0:
                        neighs[a].append(torch.from_numpy(np.array(b)).to(device))
            neighs_list.append(neighs)
            initial_feature = torch.FloatTensor(preprocessing.scale(graph_features_list[i]))
            if args.dataset in ['Sub_Flckr', 'Sub_Yelp']:
                aggregated_feature = sgc_precompute(initial_feature, adj, args.aggregation_times)
                reformed_futures_list.append(aggregated_feature.to(device))
            else:
                reformed_futures_list.append(initial_feature.to(device))
        return reformed_futures_list, reformed_labels_list, neighs_list, adjs_list

    val_graphs_features_list, val_graphs_labels_list, val_neighs_list, val_adj_list = feature_label_generator(val_graphs_index)
    test_graphs_features_list, test_graphs_labels_list, test_neighs_list, test_adj_list = feature_label_generator(test_graphs_index)

    config = [('linear', [args.hidden, num_attri]),
              ('linear', [label_dim, args.hidden])]

    if args.dataset in ['Sub_Flckr', 'Sub_Yelp']:
        config_chemi = [('linear', [args.hidden, label_dim]),  
                        ('leaky_relu', [args.hidden, args.hidden])]
    else:
        config_chemi = [('linear', [args.hidden, label_dim + args.hidden]),  
                        ('leaky_relu', [args.hidden, args.hidden])]

    config_scal = [('linear', [args.hidden * (num_attri + 1) + label_dim * (args.hidden + 1), args.hidden])]

    config_trans = [('linear', [args.hidden * (num_attri + 1) + label_dim * (args.hidden + 1), args.hidden])]

    if args.dataset in ['Sub_Flckr', 'Sub_Yelp']:
        model = Meta_NA
    else:
        model = Meta_MSNA

    inductive_meta = model(config, config_chemi, config_scal, config_trans, args, num_attri, label_dim).to(device)

    patience = 20
    early_stopping = EarlyStopping(args.dataset, patience, verbose=True)

    val_x_spt, val_y_spt, val_x_qry, val_y_qry, val_idx_spt, val_idx_qry \
        = task_data_generator(val_graphs_features_list, val_graphs_labels_list, device)

    test_x_spt, test_y_spt, test_x_qry, test_y_qry, test_idx_spt, test_idx_qry \
        = task_data_generator(test_graphs_features_list, test_graphs_labels_list, device)

    for Epoch in range(args.epoch):
        inductive_meta.train()
        train_graphs_features_list, train_graphs_labels_list, train_neighs_list, train_adj_list = feature_label_generator(train_graphs_index, shuffle=True)

        train_x_spt, train_y_spt, train_x_qry, train_y_qry, train_idx_spt, train_idx_qry \
            = task_data_generator(train_graphs_features_list,
                                  train_graphs_labels_list, device)

        acc, loss, microf1 = inductive_meta.forward(train_x_spt, train_y_spt, train_x_qry,
                                                    train_y_qry,
                                                    train_idx_spt, train_idx_qry,
                                                    train_graphs_features_list,
                                                    train_neighs_list,
                                                    args.l2_coef,
                                                    Lab,
                                                    update_step=args.update_step,
                                                    batch_size=args.batch_size,
                                                    training=True)
        if Epoch % 10 == 0:
            print('Step:', Epoch,
                  '\t Meta_Training_Accuracy:{:.4f},loss:{:.4f}, micro-f1:{:.4f}'.format(acc, loss, microf1))
        inductive_meta.eval()
        val_acc, val_loss, val_microf1 = inductive_meta.forward(val_x_spt, val_y_spt, val_x_qry,
                                                                val_y_qry,
                                                                val_idx_spt, val_idx_qry,
                                                                val_graphs_features_list,
                                                                val_neighs_list,
                                                                args.l2_coef,
                                                                Lab,
                                                                update_step=args.update_step,
                                                                batch_size=args.batch_size,
                                                                training=False)

        if Epoch % 10 == 0:
            print(
                '\t Meta_validating_Accuracy:{:.4f},loss:{:.4f}, micro-f1:{:.4f}'.format(val_acc, val_loss, val_microf1))
        early_stopping(val_acc, inductive_meta)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    test_model = model(config, config_chemi, config_scal, config_trans, args, num_attri, label_dim).to(device)
    test_model.load_state_dict(torch.load('./meta_checkpoint.pkl'.format(args.dataset)))
    test_model.eval()
    test_acc, test_loss, test_microf1 = test_model.forward(test_x_spt, test_y_spt, test_x_qry,
                                                           test_y_qry,
                                                           test_idx_spt, test_idx_qry,
                                                           test_graphs_features_list,
                                                           test_neighs_list,
                                                           args.l2_coef,
                                                           Lab,
                                                           update_step=args.update_step,
                                                           batch_size=args.batch_size,
                                                           training=False)
    print('\t Testing_Accuracy:{:.4f},loss:{:.4f}, micro-f1:{:.4f}'.format(test_acc, test_loss, test_microf1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cuneiform', help='Dataset to use, including, Sub_Flickr, Sub_Yelp, Cuneiform, COX2, DHFR.')
    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--hidden', type=str, default=16, help='number of hidden neurons for gnn')
    parser.add_argument('--epoch', type=int, default=201, help='epoch number')
    parser.add_argument('--task_lr', type=float, default=0.5, help='task level adaptation learning rate, for Cuneiform, Sub_Flickr, Sub_Yelp, 0.5, others, 0.005')
    parser.add_argument('--l2_coef', type=float, default=0.001, help='l2 regularization coefficient, for Flickr, 1, others, 0.001')
    parser.add_argument('--meta_lr', type=float, default=0.01, help='the outer framework learning rate')
    parser.add_argument('--beta_hidden', type=int, default=16, help='number of hidden neurons for gnn')
    parser.add_argument('--batch_size', type=int, default=10, help='number of graphs per batch')
    parser.add_argument('--update_step', type=int, default=2, help='number of task level adaptation steps')
    parser.add_argument('--seed', type=int, default=16, help='random seed')

    args = parser.parse_args()

    main(args)
