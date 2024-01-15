import math
import torch
import torch.nn as nn
from .tft import TargetTaskFeatureTuner

class SmallEcapaTdnn(nn.Module):
    def __init__(self, ssl_hidden_layer, ssl_hidden_size, embedding_size, weighted_sum=False, use_TFT=False):
        super(SmallEcapaTdnn, self).__init__()

        C = 512
        self.norm = nn.InstanceNorm1d(ssl_hidden_size)
        self.weighted_sum = weighted_sum
        if weighted_sum:
            self.w = nn.Parameter(torch.rand(1, ssl_hidden_layer + 1, 1, 1))

        self.conv1 = nn.Conv1d(ssl_hidden_size, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        self.layer1 = Bottle2neck(C, C, 3, 2, 8)
        self.layer2 = Bottle2neck(C, C, 3, 3, 8)
        self.layer3 = Bottle2neck(C, C, 3, 4, 8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)
        
        self.use_TTT = use_TFT
        if use_TFT:
            self.ttt = TargetTaskFeatureTuner(ssl_hidden_layer, ssl_hidden_size)
            
    def forward(self, x):
        # weighted-sum or TTT
        if self.weighted_sum:
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)
        elif self.use_TTT:
            x = self.ttt(x)
        elif len(x.size()) == 4:
            x = x[:, -1, :, :]

        # norm
        x = x.permute(0, 2, 1)
        x = self.norm(x)
           
        # ecapa
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.layer4(x)
        x = self.relu(x)

        time = x.size()[-1]

        temp1 = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, time)
        temp2 = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, time)
        gx = torch.cat((x, temp1, temp2), dim=1)
        
        w = self.attention(gx)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
    
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1

        bns = []
        convs = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_split = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = x_split[i] if i == 0 else sp + x_split[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), 1)
        x = torch.cat((x, x_split[self.nums]), 1)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        
        x = self.se(x)
        x += identity
        return x 
