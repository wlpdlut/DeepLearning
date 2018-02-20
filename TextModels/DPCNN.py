# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:24:15 2018

@author: liping
"""
import torch.nn as nn
from models.LockDrop import LockDrop
from collections import OrderedDict

class DPCNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,cout,linear_dim,
                      kernel_size,num_classes,conv_blocks=2):
        super(DPCNN,self).__init__()
        
        self.embeding = nn.Embedding(vocab_size,embed_dim)
        
        self.convs = nn.Sequential( OrderedDict([
                               ( 'drop01', LockDrop(p=0.4) ),
                               ( 'conv_01', nn.Conv1d(embed_dim,cout,kernel_size=kernel_size,padding=1)),
                               ( 'nb01',   nn.BatchNorm1d(cout) ),
                               ( 'prelu01',nn.PReLU() ),
                               ( 'drop02', LockDrop(p=0.1) ),
                                
                               ( 'conv_11', nn.Conv1d(cout,cout,kernel_size=kernel_size,padding=1) ),
                               ( 'nb11',   nn.BatchNorm1d(cout) ),
                               ( 'prel11',nn.PReLU() ),
                               ( 'drop11', LockDrop(p=0.1) )]) 
                                ) 
        
        self.shape_match = nn.Sequential( OrderedDict([
                ( 'conv_21', nn.Conv1d(embed_dim,cout,kernel_size=1) ),
                ( 'prelu21',nn.PReLU() ),
                ( 'drop21', LockDrop(p=0.1) )])
                )
        
        self.max_pool = nn.MaxPool1d(kernel_size=3,stride=2)
        
        self.dpcnn_blocks = nn.ModuleList(
                      [nn.Sequential( OrderedDict([
                                      ('conv_00'+str(i),nn.Conv1d(cout,cout,kernel_size=kernel_size,padding=1)),
                                      ('nb_00'+str(i),nn.BatchNorm1d(cout) ),
                                      ('prelu_00'+str(i),nn.PReLU() ),
                                      ('drop_00'+str(i),LockDrop(p=0.1) ),
                                      
                                      ('conv_11'+str(i),nn.Conv1d(cout,cout,kernel_size=kernel_size,padding=1) ),
                                      ('nb_11'+str(i),nn.BatchNorm1d(cout) ),
                                      ('prelu_11'+str(i),nn.PReLU() ),
                                      ('drop_11'+str(i),LockDrop(p=0.1))]) )  
                        for i in range(conv_blocks)]
                      )
                      
                      
        self.fc = nn.Sequential(
                 nn.Linear(cout,linear_dim),
                 nn.BatchNorm1d(linear_dim),
                 nn.PReLU(),
                 nn.Dropout(p=0.25),
                 nn.Linear(linear_dim,num_classes)
                 )
    
#    def get_optimizer(self,lr1=0.002,lr2=0.001,weight_decay = 0):
#        embeds = list(self.embeding.parameters())
#       
#        others = [ param_ for name_,param_ in self.named_parameters() 
#                              if name_.find('embeding')==-1]
#
#        optimizer = torch.optim.Adam([
#                dict(params=embeds, weight_decay=weight_decay, lr=lr1),
#                dict(params=others, weight_decay=weight_decay, lr=lr2)
#            ], lr=0.001)
#        
#        return optimizer
#    
#    
#    def l2_regularized(self,reg=0.0001):
#        '''
#        only for convs
#        '''
#        reg_loss = 0.0
#        for name_,param_ in self.named_parameters():
#            if name_.find('conv_')==-1:
#                continue
#            #print(name_)
#            reg_loss += reg * torch.sum(param_ * param_)
#        return reg_loss
    
    
    def forward(self,x):
        #[B,L] ->[B,L,D]
        embeds = self.embeding(x)
    
        #[B,L,D] -> [B,D,L]
        embeds = embeds.permute(0,2,1)
        
        # 2 conv blocks
        outs = self.convs(embeds)
            
        #[B,D,L] -> [B,D,L]
        shaped_embeds = self.shape_match(embeds)
        outs = shaped_embeds+outs
        
        #[B,D,L] -> [B,D,X]
        # dpcnn blocks
        for dpm in self.dpcnn_blocks:
            pools = self.max_pool(outs)
            outs = dpm(pools)
            outs = outs+pools
            
        
        #[B,D,X] -> [B,D,1]
        # as unknown size ,so use function instead of module
        outs = nn.functional.max_pool1d(outs,outs.size(2))
        
        # [B,D,1] -> [B,D]
        outs = outs.squeeze(dim=2)
        
        # linear
        outs = self.fc(outs)
        
        #y = nn.functional.sigmoid(outs)
        return outs