import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CCE(nn.Module):
    def __init__(self, num_class, embedding_size=None, class_weight=None):
        super(CCE, self).__init__()
        self.is_embed_input = embedding_size is not None

        if class_weight is not None and type(class_weight) is list:
            class_weight = torch.FloatTensor(class_weight)
        if self.is_embed_input:
            self.fc = nn.Linear(embedding_size, num_class)
        
        self.cce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        if self.is_embed_input:
            x = self.fc(x)
        
        if label is not None:
            loss = self.cce(x, label)
            return loss
        else:
            return x
        
class AAMSoftmax(nn.Module):
    '''AAMSoftmax classification loss for speaker verification. 
    see here -> 'In defence of metric learning for speaker recognition'
    '''
    def __init__(self, embedding_size, num_class, margin, scale, class_weight=None, topk_panelty=None):
        super(AAMSoftmax, self).__init__()
        
        self.scale = scale
        
        # FC
        self.fc = torch.nn.Parameter(torch.FloatTensor(num_class, embedding_size), requires_grad=True)
        nn.init.xavier_normal_(self.fc, gain=1)
        
        # CE
        if class_weight is not None:
            if type(class_weight) is list:
                class_weight = torch.tensor(class_weight)
            self.ce = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.ce = nn.CrossEntropyLoss()
        
        # positive_margine
        self.cos_pos_m = math.cos(margin)
        self.sin_pos_m = math.sin(margin)

        # topk panalty
        self.topk = None
        if topk_panelty is not None:
            self.topk, topk_margin = topk_panelty
            self.cos_neg_m = math.cos(topk_margin)
            self.sin_neg_m = math.sin(topk_margin)
            
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.fc))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        
        # cos(theta + m) -> utilized angle addition and subtraction formulas
        phi = cosine * self.cos_pos_m - sine * self.sin_pos_m
    
        # cos(theta - m)
        if self.topk is not None:
            panalty_cos = cosine * self.cos_neg_m + sine * self.sin_neg_m
        
        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        if self.topk is not None:
            # except self
            idx_map = torch.ones(cosine.size(), device=cosine.device)
            idx_map.scatter_(1, label.view(-1, 1), -2)
            negative_idx = cosine * idx_map
            
            # select topK index (panalty_idx)
            K = self.topk
            topk = negative_idx.topk(K, dim=1)[1]
            panalty_idx = torch.zeros_like(negative_idx)           
            panalty_idx.scatter_(1, topk.view(-1, K), 1)
            
            # calculate loss
            remainder = 1 - one_hot - panalty_idx
            output = (one_hot * phi) + (panalty_idx * panalty_cos) + (remainder * cosine)
            output = output * self.scale
            
            loss = self.ce(output, label)
            
            return loss
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.scale
            
            loss = self.ce(output, label)
            
            return loss
        
# One-Class Softmax: only for one class
class OCSoftmax(nn.Module):
    def __init__(self, embedding_size, num_class=1, r_real=0.9, r_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.embedding_size = embedding_size
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        
        self.weight = nn.Parameter(torch.FloatTensor(num_class, embedding_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2,1,1e-5).mul_(1e5)
        
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
            
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        scores = x @ w.transpose(0, 1)
        if label is None:
            scores = 1 - scores # Caution
            return scores

        scores[label == 0] = self.r_real - scores[label == 0]
        scores[label == 1] = scores[label == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()
        return loss

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, output_s: torch.Tensor, output_t) -> torch.Tensor:
        output_s = self.softmax(output_s)
        output_t = self.softmax(output_t)
        return self.kldiv(output_s.log(), output_t)