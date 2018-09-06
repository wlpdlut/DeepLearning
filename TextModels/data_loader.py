# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:17:55 2017

@author: liping
"""

import torch
from torch.utils.data import Dataset,TensorDataset
import nltk
import random

#from collections import defaultdict
#from nltk.stem import PorterStemmer
#stemmer = PorterStemmer()

data_dir = r'D:\eclipseworkspace\Kaggle\ToxicComments\data\toxiccoments-kaggle-2'


def is_ascii(word):
    for ch in word:
        if ch<'a' or ch>'z':
            return False
    return True

   
class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<PAD>':0,'<UNK>':1}
        self.idx2word = {0:'<PAD>',1:'<UNK>'}
        
        
    def add_word(self, word):
        if word not in self.word2idx: 
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
      
    def __len__(self):
        return len(self.word2idx)
    
class Corpus(object):
    def __init__(self,sources):
        self.dictionary = Dictionary()
        self.tokenized(sources)        
        
    def tokenized(self,sources):
        for line in sources:
            if not isinstance(line,str):
                continue
            
            words = nltk.word_tokenize(line)
            for word in words:
                self.dictionary.add_word(word)
                 
    def get_vocab_size(self):
        return len(self.dictionary)
     
        
class MultiClassDataSet(Dataset):
    '''
    multi label classify
    binary cross entrophy
    '''
    def __init__(self,corpus,sources,targets=None,max_len=100):
        super(MultiClassDataSet,self).__init__()
        self.corpus = corpus
        self.targets = targets
        self.max_len = max_len 
        self.tensors = []
        self.lengths = []           
        self._tensors(sources)
        
    def _tensors(self,sources):
        cnts = 1
        for sents in sources:
            cnts += 1
            try:
                words = nltk.word_tokenize(sents)
            except Exception as e:
                words = ['cxyz']
                
            sstensor = torch.LongTensor(self.max_len)
            sstensor.fill_(0)
            cur_idx = 0
            for word in words:  
                if cur_idx >= self.max_len:
                    break

                if word in self.corpus.dictionary.word2idx:
                    sstensor[cur_idx] = self.corpus.dictionary.word2idx[word]
                else:
                    sstensor[cur_idx] = self.corpus.dictionary.word2idx['<UNK>']
                
                cur_idx += 1
                    
            self.tensors.append(sstensor)
            self.lengths.append(cur_idx if cur_idx<self.max_len else self.max_len)
             
            if cnts%500 == 0:
                print(cnts)
            
    def __getitem__(self,idx): 
        if self.targets is not None:
            return self.tensors[idx],self.targets[idx] if self.targets is not None else None
        else:
            return self.tensors[idx]
        
    def __len__(self):
        return len(self.tensors)
    
    def split(self):
        length = len(self.tensors)
        tensors = torch.cat(self.tensors,0).view(-1,100)
        
        eval_idx = random.sample(range(length),int(0.5*length) )
        ds_eval = TensorDataset(tensors[eval_idx],self.targets[eval_idx])
        
        train_idx = list(set(range(length)) - set(eval_idx))
        ds_train = TensorDataset(tensors[train_idx],self.targets[train_idx])
        
        return ds_eval,ds_train
        
class TextDataSet(Dataset):
    '''
    Dataset for classify
    prepare to deprecate it in future
    '''
    def __init__(self,corpus,sources,targets,max_len=32,sep=' '):
        self.sources = sources
        self.targets = targets
        self.max_len = max_len
        self.sep = sep
        self.corpus = corpus
        
    def __getitem__(self,idx):
        sstensor = torch.LongTensor(self.max_len)
        sstensor.fill_(0)
        line = self.sources[idx]
        for tidx,word in enumerate(line.strip('\n').split(self.sep)):
            if tidx >= self.max_len:
                break
            if word in self.corpus.dictionary.word2idx:
                sstensor[tidx] = self.corpus.dictionary.word2idx[word]
            else:
                sstensor[tidx] = self.corpus.dictionary.word2idx['<UNK>']
                
        return sstensor,torch.LongTensor([int(self.targets[idx])])
    
    def __len__(self):
        return len(self.sources)
 

    
class LMDataSet(Dataset):
    '''
    Dataset for Language Model
    '''
    def __init__(self,corpus,sources,sep=' '):
        self.sep = sep
        self.corpus = corpus 
        self.sources = self._source_tensors(sources)
        
    def _source_tensors(self,sources):
        
        tensors = []
        for line in sources:
            for word in line.strip('\n').split(self.sep):
                if word in self.corpus.dictionary.word2idx:
                    tensors.append(self.corpus.dictionary.word2idx[word])
                   
                else:
                    tensors.append(self.corpus.dictionary.word2idx['<UNK>'])
                    
        return torch.LongTensor(tensors)
        
        
    def __getitem__(self,idx):
        return self.sources[idx],self.sources[idx+1]
    
    def __len__(self):
        return len(self.sources)-1
 

