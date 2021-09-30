import torch
import torch.nn.functional as F
from torch import nn


class LinearSVD(nn.Module):
    """A sparse variational dropout apply to linear layer.
    """
    def __init__(self, in_features, out_features, threshold=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self._bias = bias

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        if self._bias: self.bias = nn.Parameter(torch.Tensor(1, out_features))
        
        self._init_weights()

    def _init_weights(self):
        if self._bias: self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            if self._bias: lrt_mean =  F.linear(x, self.W) + self.bias
            else: lrt_mean =  F.linear(x, self.W)
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        if self._bias: return F.linear(x, self.W * (self.log_alpha < self.threshold).float()) + self.bias
        else: return F.linear(x, self.W * (self.log_alpha < self.threshold).float()) 
        
    @property
    def kl_loss(self):
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        kl = - torch.sum(kl)
        return kl