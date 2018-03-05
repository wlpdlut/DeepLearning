import torch
from torch import nn

class FastText(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,linear_hidden_dim,num_classes):
        super(FastText, self).__init__()
        
        self.embed_dim = embedding_dim
        self.pre = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim*2),
            nn.BatchNorm1d(embedding_dim*2),
            nn.ReLU()
        )
       
        self.embeding = nn.Embedding(vocab_size,embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*2,linear_hidden_dim),
            nn.BatchNorm1d(linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim,num_classes)
        )
      
    def forward(self,x):
        #[B,L] -> [B,L,D]
        embeds = self.embeding(x)
        B,L,_ = embeds.size()
        #[B,L,D] -> [B,L,2D]
        content = self.pre(embeds.contiguous().view(-1,self.embed_dim)).view(B,L,-1)
        #[B,L,2D] ->[B,2D]
        content = torch.mean(content,dim=1)
        out=self.fc(content)
        #
        #out = nn.functional.sigmoid(out)
        return out
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 