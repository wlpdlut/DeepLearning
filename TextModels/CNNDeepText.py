# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 21:12:28 2018

@author: liping
"""

import torch 
from torch import nn
from models.LockDrop import LockDrop

kernel_sizes =  [1,2,3,4]
class CNNDeepText(nn.Module): 
    def __init__(self, vocab_size,embeding_dim,
                 cout,linear_hidden_dim,num_classes,seq_len=100):
        super(CNNDeepText, self).__init__()
        
        self.encoder = nn.Embedding(vocab_size,embeding_dim)
        self.drop = LockDrop(p=0.4)

        convs = [ nn.Sequential(
                                nn.Conv1d(in_channels = embeding_dim,
                                        out_channels = cout,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(cout),
                                nn.ReLU(),

                                nn.Conv1d(in_channels = cout,
                                        out_channels =cout,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(cout),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size = (seq_len - kernel_size*2 + 2))
                            )
            for kernel_size in kernel_sizes ]

    
        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*(cout),linear_hidden_dim),
            nn.BatchNorm1d(linear_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_dim,num_classes)
        )
        

      

    def forward(self, x):
        embeds = self.encoder(x)
        embeds = self.drop(embeds)
        
        out = [ conv(embeds.permute(0,2,1)) for conv in self.convs ]
        out = torch.cat(out,dim=1)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        #
        #logits = nn.functional.sigmoid(logits)
        return logits