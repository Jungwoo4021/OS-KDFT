import torch
import torch.nn as nn
from .tft import TargetTaskFeatureTuner

class LinearClassifier(nn.Module):
    def __init__(self, ssl_hidden_layer, ssl_hidden_size, embedding_size, weighted_sum=False, use_TFT=False):
        super(LinearClassifier, self).__init__()
        assert not weighted_sum or not use_TFT, 'weighted sum and TFT cannot be used simultaneously'
        
        self.weighted_sum = weighted_sum

        self.fc1 = nn.Linear(ssl_hidden_size * 2, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)

        if self.weighted_sum:
            self.w = nn.Parameter(torch.rand(ssl_hidden_layer + 1, 1, 1))

        self.use_TFT = use_TFT
        if use_TFT:
            self.TTT = TargetTaskFeatureTuner(ssl_hidden_layer, ssl_hidden_size)
            
    def forward(self, x):
        # x shape: (bs, num_hidden_layer, sequence_length, hidden_size)
        if self.weighted_sum:
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)
        elif self.use_TFT:
            x = self.TTT(x)
        elif len(x.size()) == 4:
            x = x[:, -1, :, :]

        x = x.permute(0, 2, 1)
        x = torch.cat((x.mean(dim=-1), x.std(dim=-1)), dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        
        return x