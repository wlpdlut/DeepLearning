# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:10:14 2018

@author: liping
"""

import torch
from torch import nn
from collections import OrderedDict
from models.LockDrop import LockDrop


class Inception(nn.Module):
    def __init__(self,cin,co,relu=True,norm=True):
        super(Inception, self).__init__()
        assert(co%4==0)
        cos=[int(co/4)]*4
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm1d(co))
        if relu:self.activa.add_module('relu',nn.ReLU())
        self.branch1 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[0], 1,stride=1)),
            ])) 
        self.branch2 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU()),
            ('conv3', nn.Conv1d(cos[1],cos[1], 3,stride=1,padding=1)),
            ]))
        self.branch3 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[2], 3,padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU()),
            ('conv3', nn.Conv1d(cos[2],cos[2], 5,stride=1,padding=2)),
            ]))
        self.branch4 =nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin,cos[3], 3,stride=1,padding=1)),
            ]))
    
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=self.activa( torch.cat((branch1,branch2,branch3,branch4),1) )
        return result
    
    
class CNNText(nn.Module):
    def __init__(self, vocab_size,embeding_dim,hidden_dim,incept_dim,
                 linear_hidden_size,num_classes,seq_len):
        super(CNNText, self).__init__()
 
        self.embeding = nn.Embedding(vocab_size,embeding_dim)
        self.drop = LockDrop(p=0.4)
        
        self.conv=nn.Sequential(
            Inception(embeding_dim,incept_dim),
            Inception(incept_dim,incept_dim),
            nn.MaxPool1d(seq_len)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(incept_dim,linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size,num_classes)
        )
      
 
    def forward(self,x):
        '''
        x ->[B,L]
        '''
        emebds=self.embeding(x)
        emebds = self.drop(emebds)
        
        out=self.conv(emebds.permute(0,2,1))
        out = out.view(out.size(0),-1)
        out=self.fc(out)
        #
        #out = nn.functional.sigmoid(out)
        return out
        