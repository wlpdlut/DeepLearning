# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:47:55 2018

@author: liping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,size,num_layers,fn=F.tanh):
        super(Highway,self).__init__()
        self.fn = fn
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size,size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size,size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size,size) for _ in range(num_layers)])
        
    def forward(self,x):
        '''
        x --> [B,size]
        '''
        for l  in range(self.num_layers):
            nonlinear = self.fn( self.nonlinear[l](x) )
            gate = F.sigmoid( self.gate[l](x) )
            linear = self.linear[l](x)
            
            x = gate*nonlinear + (1-gate)*linear
            
        return x

if __name__ == '__main__':
    from torch.autograd import Variable
    x = Variable(torch.randn(50,30))
    hi = Highway(30,20)
    print(len(list(hi.parameters())))
    out = hi(x)
    print(out.size())