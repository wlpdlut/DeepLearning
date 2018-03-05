# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:44:38 2018

@author: liping
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class LanguageModel(nn.Module):
    def __init__(self,vocab_len,embed_dim,hid_dim,batch_size,num_layers=1,dropout=0.2):
        super(LanguageModel,self).__init__()
        
        self.embed = nn.Embedding(vocab_len,embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(embed_dim,hid_dim,num_layers=num_layers)
        self.fc = nn.Linear(hid_dim,vocab_len)
        
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.num_layer = num_layers
        
        self.init_weights()
        
        
    def forward(self,x,hidden):
        #[S,B] -- > [S,B,D]
        embeds = self.embed(x)
        embeds = self.dropout(embeds)
        
        #[S,B,D] -> [S,B,HID_DIM]
        out,hidden = self.rnn(embeds,hidden)
        
        #[S,B,H] -> [S*B,H] -> [S*B,VOCAB_LEN]
        decodes = self.fc(out.view(-1,self.hid_dim))
        decodes = decodes.view( out.size(0), out.size(1),-1)
        
        return decodes,hidden
    
    def init_weights(self,init_range=0.1):
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range,init_range)
        
    def init_hidden(self):
        return  Variable(torch.zeros(self.num_layer,self.batch_size,self.hid_dim))