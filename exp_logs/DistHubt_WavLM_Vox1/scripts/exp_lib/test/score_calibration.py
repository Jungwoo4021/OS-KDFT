from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

class QMF(nn.Module):
    def __init__(self, input_dim, num_iter=50, converged_loss=1e-4, lr=1e-2):
        super(QMF, self).__init__()
        self.num_iter = num_iter
        self.converged_loss = converged_loss
        
        # linear model
        self.fc = nn.Linear(input_dim, 1)
        nn.init.constant_(self.fc.weight, 1.0 / input_dim)
        nn.init.constant_(self.fc.bias, 0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        y = self.fc(x)
        return y

    def negative_log_sigmoid(self, lodds):
        """-log(sigmoid(log_odds))"""
        return torch.log1p(torch.exp(-lodds))

    def cllr(self, target_llrs, nontarget_llrs):
        '''Calculate the CLLR of the scores
        '''
        return 0.5 * (torch.mean(self.negative_log_sigmoid(target_llrs)) + torch.mean(self.negative_log_sigmoid(-nontarget_llrs)))/np.log(2)
    
    def train(self, train_set):
        best_loss = 1000000.0
        with tqdm(total = self.num_iter, ncols = 100) as pbar:
            for _ in range(self.num_iter):
                new_target_llrs = []
                new_nontarget_llrs = []
                for x, label in train_set:
                    self.optimizer.zero_grad()
                    x = torch.tensor(x).to(torch.float32)
                    
                    x = self(x)
                    if label == 0:
                        print(label, x)
                        new_nontarget_llrs.append(x.view(-1))
                    elif label == 1:
                        print(label, x)
                        new_target_llrs.append(x.view(-1))
                print(len(new_target_llrs), len(new_nontarget_llrs))
                new_target_llrs = torch.cat(new_target_llrs)
                new_nontarget_llrs = torch.cat(new_nontarget_llrs)
                loss = self.cllr(new_target_llrs, new_nontarget_llrs)
                
                description = 'loss: {}'.format(loss.item())
                loss.backward()
                self.optimizer.step()

                if (best_loss - loss < self.converged_loss):
                    break
                else:
                    if loss < best_loss:
                        best_loss = loss

            pbar.set_description(description)
            pbar.update(1)