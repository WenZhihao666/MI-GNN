import numpy as np
import torch
from random import sample



def task_data_generator(features, labels, device):
    x_spt = []
    y_spt = []
    idx_spt = []
    x_qry = []
    y_qry = []
    idx_qry = []
    for t in range(len(features)):
        train_index = sample(range(len(features[t])), int(0.5 * len(features[t])))
        test_index = list(set(range(len(features[t]))) - set(train_index))
        train_attr = (features[t])[train_index]
        test_attr = (features[t])[test_index]
        train_label = (labels[t])[train_index]
        test_label = (labels[t])[test_index]
        x_spt.append(train_attr.to(device))
        y_spt.append(train_label.to(device))
        idx_spt.append((torch.from_numpy(np.array(train_index)).to(device)))
        x_qry.append(test_attr.to(device))
        y_qry.append(test_label.to(device))
        idx_qry.append((torch.from_numpy(np.array(test_index)).to(device)))

    return x_spt, y_spt, x_qry, y_qry, idx_spt, idx_qry
