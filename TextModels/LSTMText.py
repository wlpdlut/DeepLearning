from torch import nn
from models.LockDrop import LockDrop


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMText(nn.Module): 
    def __init__(self, vocad_size,embed_dim,hidden_dim,num_layers,linear_hidden_dim,kmax,num_classes):
        super(LSTMText, self).__init__()
        
        self.kmax = kmax
        self.num_classes = num_classes
        
        self.embeding = nn.Embedding(vocad_size,embed_dim,padding_idx=0)
        self.dropout = LockDrop(p=0.4) #0.4#0.5
        
        self.rnn =nn.LSTM( input_size = embed_dim,
                            hidden_size = hidden_dim,
                            num_layers = num_layers,
                            bias = True,
                            batch_first = True,
                            bidirectional = True)
       
        
        
        self.fc = nn.Sequential(
            nn.Linear(kmax*(hidden_dim*2),linear_hidden_dim),
            nn.BatchNorm1d(linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim,num_classes)
        )
      
    def forward(self, x):
        '''
        x  ->[B,L]
        '''
        #->[B,L,D]
        embeds = self.embeding(x)
        embeds = self.dropout(embeds)
        
        #[B,L,D] -> [B,D,L]
        out = self.rnn(embeds)[0].transpose(2,1)
        #[B,D,L] ->[B,D,K]
        out = kmax_pooling(out,2,self.kmax)
        #[B,D,K] -> [B,D*K]
        reshaped = out.view(out.size(0), -1)
        #->[B,class]
        logits = self.fc((reshaped))
        #
        #logits = nn.functional.sigmoid(logits)
        return logits
 

if __name__ == '__main__':
    import torch
    x =  torch.autograd.Variable( torch.arange(100).view(10,10) )
    print(x)
    lstm = LSTMText(vocad_size=100,embed_dim=100,
                    hidden_dim=100,linear_hidden_dim=100,
                    kmax=2,num_classes=3)
    lstm(x)