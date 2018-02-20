# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:54:18 2018

@author: liping
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,input_dim):
        super(Attention,self).__init__()
        
        self.tanh = nn.Tanh()
        # attention dim is the same as input
        self.linear = nn.Linear(input_dim,input_dim)
        #
        self.atten_linear = nn.Linear(input_dim,1,bias=False)
     
    def forward(self,x,pad_masked=None):
        '''
        x  ->  [L,B,D]
        pad_masked -> [L,B]
        '''
        #[L,B,D] -> [B*L,D]
        u = self.tanh( self.linear(x.view(-1,x.size(2))) )
        
        #[B*L,D]->[B*L,1]->[L,B]
        atten = self.atten_linear(u).view(x.size(0),x.size(1))
        # [L,B] -> [L,B]
        if pad_masked is not None:
            a = F.softmax( atten*pad_masked ,dim=0 )
        else:
            a = F.softmax( atten ,dim=0 )
            
        # a ->[L,B], x->[L,B,D]
        s =  a.unsqueeze(2).expand_as(x)*x #->[L,B,D]
        # [L,B,D]->[1,B,D]->[B,D]
        return s.sum(0,keepdim=True).squeeze(0)
        
  

if __name__ == '__main__':
    att = Attention(4)
    #[B,L,D]
    x = torch.randn(2,3)
    x[:,2:]=0
    print(x)
    masked = x.eq(0).float().t().contiguous()
    x = x.unsqueeze(2).expand(2,3,4)
    print(x)
    
    print( att(Variable(x.transpose(0,1).contiguous()),Variable(masked)) )
    
    #
    #attn(gru(x))
    
         
         