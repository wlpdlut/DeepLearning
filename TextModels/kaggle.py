# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:17:10 2018

@author: liping
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import os
from datautils.data_loader import Corpus,MultiClassDataSet
from torch.utils.data.dataloader import DataLoader

import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def _corpus_and_dataset():
    df_train = pd.read_csv(os.path.join(data_dir,'train.csv'))
    train = df_train['comment_text'].values
    label = df_train[[ 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']].values
        
#    corpus = Corpus(train)
#    torch.save(corpus,os.path.join(data_dir,'corpus.pkl') )
    corpus = torch.load(os.path.join(data_dir,'corpus.pkl') )
    print(len(corpus.dictionary) )
    ds_train = MultiClassDataSet(corpus,train,torch.from_numpy(label))
    torch.save(ds_train, os.path.join(data_dir,'ds_train.pkl') )
    
    df_test = pd.read_csv( os.path.join(data_dir,'test.csv') )
    df_test['comment_text'] = df_test['comment_text'].fillna('CXYZ')
    test = df_test['comment_text'].values
    
    ds_test = MultiClassDataSet(corpus,test)
    torch.save(ds_test,os.path.join(data_dir,'ds_test.pkl'))


def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()

import random
def _shuffle(x):
    try:
        zero_idx = x.index(0)
        temp = x[:zero_idx]
        random.shuffle(temp)
        return temp+x[zero_idx:]
    except:
        random.shuffle(x)
        return x
   
    
def _eval():
    model.eval() 
    res = []
    labels = []
    for test,label in dl_eval:
        test = Variable(test)
        if use_gpu:
            test = test.cuda()
       
        model.batch_size = test.size(0)
        outs = model(test)
        res.append(outs.data.cpu())
        labels.append(label)
        
    predications = torch.cat(res,0)
    labels = torch.cat(labels,0)
    assert(predications.size(1)==6)
    _roc = multi_roc_auc_score(labels.numpy(),predications.numpy())
    #print( 'eval roc={0}'.format(_roc) )
    return _roc
    
    
    
def predicate():
    dl_test = DataLoader( torch.load(os.path.join(data_dir,'ds_test.pkl')),
                         batch_size=batch_size, shuffle=False )
    model = torch.load( os.path.join(data_dir,'{0}_model_{1}.pkl'.format(model_name,midx) ) )

    model = model.cuda()
    model.eval() 
    res = []
    loops = 0
    for test in dl_test:
        test = Variable(test)
        if use_gpu:
            test = test.cuda()
            
        model.batch_size = test.size(0)
        outs = model(test)
        
        loops += 1
        if loops %200 == 0:
            print(loops)
        res.append(outs.data.cpu())
        
    res = torch.cat(res,0)
    print(res.size())
    assert(res.size(1)==6)
    
    sample_submission = pd.read_csv( os.path.join(data_dir,'sample_submission.csv') )
    sample_submission[[ 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']] = res
    sample_submission.to_csv(os.path.join(data_dir,'{0}.csv'.format(model.__module__[7:])), index=False)
    
    
def train():
    for e in range(1,20):    
        model.train()
        for batch_idx,(train,label) in enumerate(dl_train):
            train = Variable(train)
            label = Variable(label.float())
            
            if use_gpu:
                train = train.cuda()
                label = label.cuda()

            optim.zero_grad()
            model.batch_size = train.size(0)
            outs = model(train)

            loss = criterion(outs,label)
            avg_loss = loss.data[0]
            
            #L2 regualrized
            #if hasattr(model,'l2_regularized'):
            #    loss += model.l2_regularized()
            #
            loss.backward()
            #
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optim.step()
            #
            if batch_idx!=0 and batch_idx%200==0:
                if is_eval:
                    _roc = _eval()
                    print( 'epoch {0}  batch {1}  avg_loss {2:.5f} roc {3:.5f}'.format(
                        e,batch_idx,avg_loss,_roc) ) 
                    model.train()
                else:
                    print( 'epoch {0}  batch {1}  avg_loss {2:.5f}'.format(
                        e,batch_idx,avg_loss) ) 
    
        if e>=2:
            torch.save(model,os.path.join(data_dir,'{0}_model_{1}.pkl'.format(model.__module__[7:],e)) )

if __name__ == '__main__':
    from models.RnnClassifier import RnnClassifier
    from models.LSTMText import LSTMText
    from models.FastText import FastText
    from models.CNNIncetptionText import CNNText
    from models.CNNDeepText import CNNDeepText
    from models.RCNNText import RCNNText
    from models.DPCNN import DPCNN
    #
    data_dir = r'D:\eclipseworkspace\Kaggle\ToxicComments\data\toxiccoments-kaggle-2'
   
    batch_size = 64
    embed_dim = 200
    hidden_dim = 128 
    labels = 6
    learning_rate = 0.001
    use_gpu = True
    #
    is_predicate = False
    midx = 5
    model_name = 'CNNText'
    
    is_eval = False
    #_corpus_and_dataset()
    
    if is_predicate:
        predicate()
    else: 
        corpus = torch.load( os.path.join(data_dir,'corpus.pkl') ) 
        print(corpus.get_vocab_size())
        ds_train = torch.load( os.path.join(data_dir,'ds_train.pkl') )
        
        if is_eval:
            ds_eval,ds_train = ds_train.split()
            dl_eval = DataLoader(ds_eval,batch_size=batch_size,shuffle=False)
        
        print(len(ds_train))
        dl_train = DataLoader(ds_train,batch_size=batch_size, shuffle=True)
        
        # 5-> 9759
#        model = RnnClassifier(embed_dim=embed_dim,hidden_dim=hidden_dim,
#                              vocab_size=corpus.get_vocab_size(),
#                              label_size=labels,
#                              batch_size=batch_size)
        
        # iter 4, kmax=8,dropout=0.5  --->0.9775
#        model = LSTMText(vocad_size=corpus.get_vocab_size(),
#                         embed_dim=embed_dim,
#                         hidden_dim=hidden_dim,
#                         num_layers=2,
#                         linear_hidden_dim=1000,
#                         kmax=3,#8,#3,
#                         num_classes=labels)
           

        
#        model = FastText(vocab_size=corpus.get_vocab_size(),
#                         embedding_dim=embed_dim,
#                         hidden_dim=hidden_dim,
#                         linear_hidden_dim=1000,
#                         num_classes=labels)
      
        model = CNNText(vocab_size=corpus.get_vocab_size(),
                        embeding_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        incept_dim=512,
                        linear_hidden_size=1000,
                        num_classes=labels,seq_len=100)
        #9749 ->5
#        model = CNNDeepText(vocab_size=corpus.get_vocab_size(),
#                            embeding_dim=embed_dim,
#                            cout=250,
#                            linear_hidden_dim=1000,num_classes=labels)
        # 9759
#        model = RCNNText( vocab_size=corpus.get_vocab_size(),
#                          embedding_dim=embed_dim,
#                          hidden_size=hidden_dim,
#                          linear_hidden_dim=1000,
#                          cout=512,kernel_size=2,num_classes=labels,kmax=3)
        

         #9750
#        model = DPCNN(
#                vocab_size=corpus.get_vocab_size(),
#                embed_dim=300,
#                cout=64,
#                kernel_size = 3,
#                linear_dim = 256,
#                num_classes = labels
#                )
          
        #model = torch.load(os.path.join(data_dir,'{0}_model_4.pkl'.format(model.__module__[7:])))
         
        if use_gpu:
            model = model.cuda()
        
#        if hasattr(model,'get_optimizer'):
#            optim = model.get_optimizer()
#        else:
#            optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
        
        optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
        criterion = nn.BCELoss()
        
        train()
    