import random

import torch
import torch.nn as nn

class TargetTaskFeatureTuner(nn.Module):
    def __init__(self, num_hidden_layers, ssl_hidden_size):
        super(TargetTaskFeatureTuner, self).__init__()
        
        self.tune_layers = nn.ModuleList([DistTuner(ssl_hidden_size, ssl_hidden_size // 12) for _ in range(num_hidden_layers)])
        
        self.att_layers = nn.ModuleList([self.init_att_layer(ssl_hidden_size) for _ in range(num_hidden_layers)])
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x[:, 1:, :, :] # B, L, S, H

        # projection
        x = torch.stack([self.tune_layers[i](x[:, i, :, :]) for i in range(x.size(1))], dim=1)
        x = x.permute(0, 1, 3, 2) # B, L, H, S
        
        # layer-wise attention
        w = torch.stack([self.att_layers[i](x[:, i, :, :]) for i in range(x.size(1))], dim=1)
        w_l = self.softmax(w)
        x = x * w_l
        x = x.sum(dim=1)
        
        # time-wise attention
        w_t = self.sigmoid(w.sum(dim=1))
        x = x * w_t
        
        x = x.permute(0, 2, 1) # B, S, H

        return x
    
    def init_att_layer(self, in_size):
        layer = self.att_layers = nn.Sequential(
            nn.Conv1d(in_size, in_size // 8, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(in_size // 8),
            nn.Tanh(),
            nn.Conv1d(in_size // 8, in_size, kernel_size=1),
        )
        return layer
    
class DistTuner(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(DistTuner, self).__init__()
        
        self.down = nn.Linear(in_size, hidden_size)
        self.gelu = nn.GELU()
        self.up = nn.Linear(hidden_size, in_size)
    
    def forward(self, x):
        # B, T, H
        x = self.down(x)
        x = self.gelu(x)
        if self.training:
            x = x + torch.normal(mean=0, std=random.uniform((x.abs().mean() * 0.1).item(), (x.abs().mean() * 0.5).item()), size=x.size(), device=x.device, dtype=x.dtype)
        x = self.up(x)
        return x