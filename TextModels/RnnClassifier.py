# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 20:36:21 2017

@author: liping
"""

import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F
from models.Attention import Attention
from models.LockDrop import LockDrop

class RnnClassifier(nn.Module):
    def __init__(self,embed_dim,hidden_dim,vocab_size,label_size,batch_size):
        super(RnnClassifier,self).__init__()
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_direction = 2
        self.embeding = nn.Embedding(vocab_size,embed_dim,padding_idx=0) 
        
        self.rnn = nn.GRU(embed_dim,hidden_dim,batch_first=True,
                           num_layers=self.num_layers,
                           bidirectional=True if self.num_direction==2 else False)
        
     
        self.hidden2label = nn.Linear(self.num_direction*hidden_dim,label_size)
        self.dropout = LockDrop(p=0.4)
        
        self.attenion = Attention(self.num_direction*hidden_dim)


    def forward(self,sentence,lengths=None):
        pad_mask = sentence.eq(0).float()
            
        embeds = self.embeding(sentence) # B*W*D
        embeds = self.dropout(embeds)
        
#        if lengths is not None:
#            packs= torch.nn.utils.rnn.pack_padded_sequence(embeds,lengths, batch_first=True)
#            packs,_ = self.rnn(packs)
#            out,_ = torch.nn.utils.rnn.pad_packed_sequence(packs, batch_first=True)
#        else:
        
        out,_ = self.rnn(embeds)#B*W*D
        out = self.attenion(out.transpose(0,1).contiguous(),pad_mask.t())
        y = self.hidden2label(out) 
        
        #y = F.sigmoid(y)
        return y    
        
        