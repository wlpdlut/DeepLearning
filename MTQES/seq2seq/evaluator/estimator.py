import torch.nn.functional as F
import torch.nn as nn


class QEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(QEstimator, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim, bias=True, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1))

    def forward(self, x):
        """
        :param x: B*L*D
        """
        # ->[B,2,D]
        hidden = F.relu(self.rnn(x)[1].transpose(0, 1).contiguous())

        # [B,2,D] -> [B,2*D]
        reshaped = hidden.view(hidden.size(0), -1).contiguous()

        # [B,1]
        return F.sigmoid(self.fc(reshaped))


