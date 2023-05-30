# based on the code of: https://github.com/lushleaf/varying-coefficient-net-with-functional-tr

import torch
import torch.nn as nn

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1] for each dimension
        :param degree: int, the degree of truncated basis
        :param knots: list of lists, the knots of the spline bases; two end points (0,1) per dimension should not be included
        """
        self.degree = degree
        self.knots = knots
        self.dim_treat = len(self.knots)
        self.num_of_basis = self.degree + 1 + len(self.knots[0])
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        out = torch.zeros(x.shape[0], self.dim_treat, self.num_of_basis)
        for d in range(self.dim_treat):
            for _ in range(self.num_of_basis):
                if _ <= self.degree:
                    if _ == 0:
                        out[:, d, _] = 1.
                    else:
                        out[:, d, _] = x[:, d]**_
                else:
                    if self.degree == 1:
                        out[:, d, _] = (self.relu(x[:, d] - self.knots[d][_ - self.degree]))
                    else:
                        out[:, d, _] = (self.relu(x[:, d] - self.knots[d][_ - self.degree - 1])) ** self.degree

        return out 


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots
        self.dim_treat = len(knots)

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis 

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d**self.dim_treat), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd,  self.d**self.dim_treat), requires_grad=True) #self.dim_treat,
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        x_feature = x[:, self.dim_treat:]
        x_treat = x[:, :self.dim_treat]
        # print(self.weight.shape, x_feature.shape)
        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T 
        x_treat_basis = self.spb.forward(x_treat) 

        if self.dim_treat ==1:
            x_treat_basis = x_treat_basis[:, 0, :]  
        elif self.dim_treat == 2:
            x_treat_basis = torch.stack([
            torch.kron(x_i[0, :], x_i[1, :]) for x_i in torch.unbind(x_treat_basis, dim=0)
                ], dim=0)
        else:
            x_treat_basis0 = torch.stack([
            torch.kron(x_i[0, :], x_i[1, :]) for x_i in torch.unbind(x_treat_basis, dim=0)
                ], dim=0)
            for i in range(self.dim_treat-2):
                x_treat_basis0 = torch.stack([
            torch.kron(x_treat_basis0[k, :], x_treat_basis[k, i+2, :]) for k in range(x_treat_basis.shape[0])
                ], dim=0)     
            x_treat_basis = x_treat_basis0
                
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        out = x_feature_weight * x_treat_basis_
        out = torch.sum(out, dim=2) 

        if self.isbias:
            # print(self.ind, self.outd, out.shape)
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)

        return out


class DCNet(nn.Module):
    def __init__(self, config):
        super(DCNet, self).__init__()

        self.cfg_representation =  [(config.dim_cov, config.num_units_representation[0], 1, 'relu'), 
                           (config.num_units_representation[0], config.num_units_representation[1], 1, 'relu')]
        self.cfg = [(config.num_units_representation[1], config.num_units_head[0], 1, 'relu'),
            (config.num_units_head[0], config.num_units_head[1], 1, 'id')]
        self.knots = [config.knots for _ in range(config.dim_treat)] 
        self.degree = config.degree
        self.dim_treat = config.dim_treat

        # representation network
        representation_blocks = []
        representation_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(self.cfg_representation):
            if layer_idx == 0:
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                representation_blocks.append(self.feature_weight)
            else:
                representation_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            representation_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                representation_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                representation_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                representation_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*representation_blocks)

        self.representation_hidden_dim = representation_hidden_dim

        # prediction head
        blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: 
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((t, hidden), 1)
        Q = self.Q(t_hidden)
        

        return Q, hidden

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()