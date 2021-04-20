import torch
from torch import nn
from torch.nn import functional as F


class Learner_1(nn.Module):
    def __init__(self, config):
        super(Learner_1, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x, neighs, vars=None):
        if vars is None:
            vars = self.vars
        neighs_features = []
        filmed_neighs_features = []
        for i in range(len(neighs)):
            neighs_features.append(x[torch.stack(neighs[i])])
            filmed_neigh_feature = torch.mean(neighs_features[i], dim=0)
            filmed_neighs_features.append(filmed_neigh_feature)
        x1 = torch.stack(filmed_neighs_features)
        x1 = F.linear(x1, vars[0], vars[1])
        neighs_features_1 = []
        filmed_neighs_features_1 = []
        for i in range(len(neighs)):
            neighs_features_1.append(x1[torch.stack(neighs[i])])
            filmed_neigh_feature_1 = torch.mean(neighs_features_1[i], dim=0)
            filmed_neighs_features_1.append(filmed_neigh_feature_1)
        x2 = torch.stack(filmed_neighs_features_1)
        x2 = F.linear(x2, vars[2], vars[3])

        return x1, x2

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
