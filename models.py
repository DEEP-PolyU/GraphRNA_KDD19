import torch
import torch.nn as nn

class pro_lstm_featwalk(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_paths, path_length, dropout, multilabel=False):
        super(pro_lstm_featwalk, self).__init__()
        target_size = nclass
        self.path_length = path_length
        self.num_paths = num_paths
        self.dropout = nn.Dropout(p=dropout)
        self.droprate = dropout
        self.multilabel = multilabel
        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(
            nn.Linear(nfeat, nhid * 4),
            nn.BatchNorm1d(nhid * 4),
            self.nonlinear,
            self.dropout,
        )
        self.mode = 'LSTM'
        if self.mode == 'LSTM':
            self.lstm = nn.LSTM(input_size=nhid * 4,
                               hidden_size=nhid,
                               num_layers=1,
                               #dropout=dropout,
                               bidirectional=True)
        elif self.mode == 'GRU':
            self.gru = nn.GRU(input_size=nhid * 4,
                              hidden_size=nhid,
                              num_layers=1,
                              #dropout=dropout,
                              bidirectional=True)

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm1d(nhid * 4),
            self.nonlinear,
            self.dropout,
            nn.Linear(nhid * 4, target_size)
        )
        #self.out = nn.Hardtanh()  # Sigmoid, Hardtanh

    def forward(self, x):  # idxinv
        x = self.first(x)
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        #_, batch_size, hiddenlen = x.size()
        #selfloop1 = torch.mean(x.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hiddenlen), dim=2)[0]

        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        _, batch_size, hidlen = outputs.size()
        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        selfloop = outputs[0]
        #selfloop1 = x[0].view(int(batch_size / self.num_paths), self.num_paths, x.size(2)).mean(dim=1)

        #weight = torch.from_numpy(np.power(self.beta, range(self.path_length))).float().cuda()
        #weight[0] = self.path_length - 1
        #outputs = torch.mean(torch.matmul(outputs.transpose_(0, 1).transpose_(1, 2), torch.diag(weight)), dim=2)
        #outputs = torch.cat((outputs, selfloop), dim=1)

        outputs = torch.cat((outputs.mean(dim=0), selfloop), dim=1)  #
        outputs = self.hidden2tag(outputs)
        return outputs
        #outputs = outputs.transpose_(1, 2).contiguous().view(senlen * num_paths, batch_size / num_paths, lablelen)
        #return torch.mean(torch.mean(outputs, dim=0),  dim=1)
    def embedding(self, x):
        x = self.first(x)
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        _, batch_size, hidlen = outputs.size()
        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        selfloop = outputs[0]
        outputs = torch.cat((outputs.mean(dim=0), selfloop), dim=1)

        return outputs