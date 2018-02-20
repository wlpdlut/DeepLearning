# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:38:11 2018

@author: liping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable

class TDNN(nn.Module):
    def __init__(self,in_chan,out_chan,kernels):
        super(TDNN,self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_chan,out_chan,k) for k in kernels])
        
    def forward(self,x):
        '''
        x : [batch_size,max_sequence_length,max_word_length,char_embed_dim]
        '''
        inp_shape = x.size()
        assert len(inp_shape)==4,'[ERROR]-input variable size must be 4'
       
        batch_size,seq_length,word_length,embed_dim = x.size()
        
        #conv1d input size should be 3
        # B = batch_size*max_seq_length
        # change to [B ,max_word_length,char_embed_dim]
        # conv1d for the word_length dimension
        x = x.view(-1,word_length,embed_dim).transpose(1,2).contiguous()
        
        #[B ,Cout,O]*len(self.convs)
        xs = [F.tanh(conv(x)) for conv in self.convs]
        
        #[B ,Cout,1]*len -> [B ,Cout]*len 
        xs = [F.max_pool1d(x,x.size(2)).squeeze(2) for x in xs]
        
        # [B ,[Cout0,Cout1...] ]
        x = torch.cat(xs,1)
        
        # return back the original size
        # B  => batch_size*seq_len
        x = x.view(batch_size,seq_length,-1)
        
        return x
        

if __name__ == '__main__':
    x = Variable(torch.FloatTensor(70,30,50,10))
    tdnn = TDNN(10,20,[2,3])
    y = tdnn(x)
    print(y.size())
          