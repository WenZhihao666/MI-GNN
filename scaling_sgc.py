import torch
import torch.nn.functional as F
from torch import nn

class Scaling(torch.nn.Module):
    def __init__(self, config_scal, args, num_attri, label_dim):
        super(Scaling, self).__init__()
        self.config = config_scal
        self.vars = nn.ParameterList()
        self.args = args
        self.num_attri = num_attri
        self.label_dim = label_dim
        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        idx = 0
        for name, param in self.config:
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'relu':
                x = F.relu(x)
            elif name is 'elu':
                x = F.elu(x)
            elif name is 'leaky_relu':
                x = F.leaky_relu(x)
        x1 = x[:self.args.hidden * self.num_attri].view(self.args.hidden, self.num_attri)
        x2 = x[self.args.hidden * self.num_attri: self.args.hidden * (self.num_attri + 1)].view(self.args.hidden)
        x3 = x[self.args.hidden * (self.num_attri + 1):
               self.args.hidden * (self.num_attri + 1) +
               self.label_dim * self.args.hidden].view(self.label_dim, self.args.hidden)
        x4 = x[self.args.hidden * (self.num_attri + 1) + self.label_dim * self.args.hidden:].view(self.label_dim)
        para_list = [x1, x2, x3, x4]
        return para_list

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
