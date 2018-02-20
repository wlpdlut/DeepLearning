# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 11:41:46 2017

@author: liping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnClassifier(nn.Module):
    def __init__(self,*args):
        super(CnnClassifier,self).__init__()
        vocab_size,embed_dim,KS,Co,labels = args
        Ci = 1
        D = embed_dim
        self.embeding = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Co,(K,D)) for K in KS])
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(len(KS)*Co,labels)
        self.static  = False
     
    def forward(self,x):
        #B*W*D
        embeds = self.embeding(x)        
        if self.static:
            embeds = torch.autograd.Variable(embeds)
        
        embeds = self.dropout(embeds)
        
        #B*1*W*D    
        embeds = embeds.unsqueeze(1)
       
        #[B*Co*W*D]*len(KS), D==1
        #[B*Co*W]*len(KS)
        X = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        #[B*Co*1]*len(KS)
        #[B*Co]*len(KS)
        X = [F.max_pool1d(sx,sx.size(2)).squeeze(2) for sx in X]
        
        #[B*Co]*len(KS) -->[B,len(KS)*Co]
        X = torch.cat(X,1)
        X = self.dropout(X)
        
        #[N*labels]
        logits = self.fc(X)
        #logits = F.sigmoid(logits)
        return logits