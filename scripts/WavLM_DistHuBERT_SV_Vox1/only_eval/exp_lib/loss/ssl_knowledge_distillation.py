import torch
import torch.nn as nn

class LastHiddenMSE(nn.Module):
    '''KD loss simply comparing the last outputs of teachers and students using MSE. 
    '''
    def __init__(self):
        super(nn.Module, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, x, label):
        return self.mse(x[:, -1, :, :], label[:, -1, :, :])

class DistilHubertKDLoss(nn.Module):
    '''SSL KD loss function used in paper 'Distilhubert: Speech representation learning by layer-wise distillation of hidden-unit bert'.
    This compute (l1_loss - cos_sim) in the hidden layers. 
    '''
    def __init__(self, cos_lambda, target_layer_idx, ssl_hidden_size):
        super(DistilHubertKDLoss, self).__init__()
        self.cos_lambda = cos_lambda
        self.num_heads = len(target_layer_idx)
        self.target_layer_idx = target_layer_idx

        self.log_sigmoid = nn.LogSigmoid()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.predict_layer = nn.ModuleList([self._make_prediction_layer(ssl_hidden_size) for _ in range(self.num_heads)])

    def forward(self, x, label):
        # prediction head
        ph = [self.predict_layer[i](x[:, -1, :, :]) for i in range(self.num_heads)]
        ph = torch.stack(ph, dim=1)
        
        # calculate L1 loss
        batch, layer, seq, hidden = ph.size()
        label = label[:, self.target_layer_idx, :, :]
        l1_loss = torch.abs(ph.reshape(batch, layer, seq * hidden) - label.reshape(batch, layer, seq * hidden))
        l1_loss = torch.mean(l1_loss, dim=-1)
        
        # calculate cosine loss
        cos_loss = self.cos_sim(ph.reshape(batch * layer * seq, hidden), label.reshape(batch * layer * seq, hidden))
        cos_loss = self.log_sigmoid(cos_loss).view(batch, layer, seq)
        cos_loss = cos_loss.sum(dim=2)

        # total loss
        loss = l1_loss - self.cos_lambda * cos_loss
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return loss
    
    def _make_prediction_layer(self, hidden_size):
        layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        return layer
    
class FitHubertKDLoss(nn.Module):
    '''SSL KD loss function used in papaer 'Fithubert: Speech representation learning by layer-wise distillation of hidden-unit bert'.
    '''
    def __init__(self, hint_lambda):
        super(FitHubertKDLoss, self).__init__()
        self.hint_lambda = hint_lambda
        
    def forward(self, x, label):
        label = label[:, 1:, :, :] # remove CNN output
        
        loss_buffer = []
        batch, layer, _, _ = x.size()
        for i in range(layer):
            # sample layer
            s_l = x[:, i, :, :].view(batch, -1)
            t_l = label[:, i, :, :].view(batch, -1)
            
            # calculate l2-loss
            l2_loss = torch.mean((s_l - t_l) ** 2, dim=-1)

            # multiply lambda to hint_loss
            l2_loss = l2_loss * self.hint_lambda if i != layer - 1 else l2_loss
            
            loss_buffer.append(l2_loss)
        
        loss = torch.mean(sum(loss_buffer))
        
        return loss