# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:58:14 2018

@author: liping
"""

import torch
from torch import nn
from models.LockDrop import LockDrop


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class RCNNText(nn.Module): 
    def __init__(self, vocab_size, embedding_dim,hidden_size,
                 linear_hidden_dim,cout,kernel_size,num_classes,kmax=2):
        super(RCNNText, self).__init__()
        self.kmax = kmax
        self.embeding = nn.Embedding(vocab_size,embedding_dim)
        self.dropout = LockDrop(p=0.4) #0.4#0.5
       

        self.rnn =nn.LSTM(input_size =embedding_dim,\
                            hidden_size = hidden_size,
                            num_layers = 2,
                            bias = True,
                            batch_first = True,
                            bidirectional = True
                            )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = 2*hidden_size+embedding_dim,
                      out_channels = cout,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(cout),
            nn.ReLU(),

            nn.Conv1d(in_channels = cout,
                      out_channels = cout,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(cout),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(kmax*cout,linear_hidden_dim),
            nn.BatchNorm1d(linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim,num_classes)
        )
        
 
    def forward(self, x):
        embeds = self.embeding(x)
        embeds = self.dropout(embeds)
        #[B,L,D] -> [B,D,L]
        out = self.rnn(embeds)[0].permute(0,2,1)
       
        out = torch.cat((out,embeds.permute(0,2,1)),dim=1)

        #[B,2D,L] -> [B,2d,Kmax]
        out = kmax_pooling(self.conv(out),2,self.kmax)
        # ->[B,2D*KMAX]
        reshaped = out.view(out.size(0), -1)
        logits = self.fc((reshaped))
        #
        #logits = nn.functional.sigmoid(logits)
        return logits
