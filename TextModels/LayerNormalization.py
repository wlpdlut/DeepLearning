# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:49:24 2018

@author: liping
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class LayerNormalization(nn.Module):
    
    def __init__(self,d_dim,eps=0.001):
        '''
        d_dim : the last dimension
        '''
        super(LayerNormalization,self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(d_dim),requires_grad=True)
        self.b = nn.Parameter(torch.zeros(d_dim),requires_grad=True)
        
    def forward(self,x):
        assert len(x.size())==2
        if x.size(1) == 1:
            return x
        
        mu = torch.mean(x,keepdim=True,dim=-1)
        gamma = torch.std(x,keepdim=True,dim=-1)
        
        # no need to expand, because of broadcast
        # expand from the last dimension always
        #out = (x-mu.expand_as(x))/(gamma.expand_as(x)+self.eps)
        
        out = (x-mu)/(gamma+self.eps)
        out = out*self.a + self.b
        
        return out
        
if __name__ == '__main__':
    x = Variable(torch.arange(20).view(5,4))
    print(x)
    ln = LayerNormalization(4)
    y = ln(x)
    print(y)
    print(y.size())