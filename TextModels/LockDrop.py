# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:03:51 2018

@author: liping
"""

from torch.autograd import Variable
import torch.nn as nn

class LockDrop(nn.Module):
    def __init__(self,p=0.5):
        super(LockDrop,self).__init__()
        self.p = p
        
    def forward(self,x):
        if not self.training or self.p==0:
            return x
        
        mask = x.data.new(1,x.size(1),x.size(2)).bernoulli_(1-self.p)/(1-self.p)
        mask = Variable(mask.expand_as(x))
        return mask*x
        
if __name__ =='__main__':
    import torch
    x = Variable(torch.randn(10,30,50))
    ld = LockDrop()
    print( ld(x) )
        