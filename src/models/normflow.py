import torch
from torch import nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN, ConditionalAutoRegressiveNN


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim, split_dim, context_dim, hidden_dim, num_layers, flow_length, 
                count_bins, order, bound, use_cuda):
        super(ConditionalNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.split_dim = split_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.flow_length = flow_length
        self.count_bins = count_bins
        self.order = order
        self.bound = bound

        self.device = 'cpu' if not use_cuda else 'cuda'
        
        self.has_prop_score = False
        self.cond_base_dist = dist.MultivariateNormal(torch.zeros(self.input_dim).float(),
                                                      torch.diag(torch.ones(self.input_dim)).float())

        self.cond_loc = torch.nn.Parameter(torch.zeros((self.input_dim, )).float())
        self.cond_scale = torch.nn.Parameter(torch.ones((self.input_dim, )).float())
        self.cond_affine_transform = T.AffineTransform(self.cond_loc, self.cond_scale)


        if self.input_dim == 1:
            self.cond_spline_nn = DenseNN(
                                        self.context_dim,
                                         [self.hidden_dim],
                                          param_dims=[self.count_bins, 
                                                        self.count_bins,
                                                      (self.count_bins - 1)]).float()
            self.cond_spline_transform = [T.ConditionalSpline(self.cond_spline_nn,
                                                            self.input_dim,
                                                             order='quadratic',
                                                             count_bins=self.count_bins,
                                                             bound=self.bound).to(self.device) for _ in range(self.flow_length)]
        else:
            self.cond_spline_nn = ConditionalAutoRegressiveNN(self.input_dim,
                                                               self.context_dim, 
                                                              [self.hidden_dim],
                                                              param_dims=[self.count_bins,
                                                                          self.count_bins,
                                                                          (self.count_bins - 1)]).float()
            self.cond_spline_transform = [T.ConditionalSplineAutoregressive(self.input_dim,
                                                                           self.cond_spline_nn,
                                                                           order='quadratic',
                                                                           count_bins=self.count_bins,
                                                                           bound=self.bound).to(self.device) for _ in range(self.flow_length)]
        self.flow_dist = dist.ConditionalTransformedDistribution(self.cond_base_dist,
                                                                      [self.cond_affine_transform] + self.cond_spline_transform) #[self.cond_affine_transform, self.cond_spline_transform]

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
            nn.ModuleList(self.transforms).cuda()
            self.base_dist = dist.Normal(torch.zeros(input_dim).cuda(),
                                         torch.ones(input_dim).cuda())
    
    def sample(self, H, num_samples=1):
        assert num_samples >= 1
        num_H = H.shape[0] if len(H.shape)==2 else 1
        dim_samples = [num_samples, num_H] if (num_samples > 1 and num_H > 1) else [num_H] if num_H > 1 else [num_samples]
        x = self.flow_dist.condition(H).sample(dim_samples)
        return x
    
    def log_prob(self, x, H):
        # x = x.reshape(-1, self.input_dim)
        cond_flow_dist = self.flow_dist.condition(H) 
        # print(x.shape, H.shape)
        return cond_flow_dist.log_prob(x)

    def model(self, X=None, H=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.cond_spline_transform))
        with pyro.plate("data", N):
                self.cond_flow_dist = self.flow_dist.condition(H)
                obs = pyro.sample("obs", self.cond_flow_dist, obs=X)
            
    def guide(self, X=None, H=None):
        pass