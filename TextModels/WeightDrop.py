# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 21:34:32 2018

@author: liping
"""

import torch
import torch.nn as nn

class WeightDrop(nn.Module):
    def __init__(self,module,weights,dropout=0.5,variational=False):
        super(WeightDrop,self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        
        
    def flatten_parameters_nop(self):
        '''
        Make flatten_parameters does nothing
        '''
        return 
    
    def _setup(self):
        # I don't understand why do such thing here
        if issubclass( type(self.module), nn.RNNBase):
            self.module.flatten_parameters = self.flatten_parameters_nop
        
        for namew in self.weights:
            w = self.module._parameters[namew]
            del self.module._parameters[namew]
            self.module.register_parameter( namew+'_raw', nn.Parameter(w.data) )
    
    def _setweights(self):
        for namew in self.weights:
            raw_w = self.module._parameters[namew+'_raw']
            
            w = nn.functional.dropout(raw_w,p=self.dropout,training=self.training)
            self.module._parameters[namew] = w
     
        
    def forward(self,*args):
        self._setweights()
        return self.module.forward(*args)
     
        
if __name__ == '__main__':
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    h0 = None

    wdrnn = WeightDrop(nn.LSTM(10, 10), ['weight_hh_l0','bias_hh_l0'], dropout=0.9)
    wdrnn = wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]
    print('Run 1:', run1)
    print('Run 2:', run2)
    