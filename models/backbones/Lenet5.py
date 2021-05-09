import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
    
    def num_flat_features(self, x):
        #バッチサイズ除く
        size = x.size()[1:]
        #全ての次元を掛け合わせる
        num_features = 1
        for s in size:
            num_features*=s
        return num_features

"""
data = torch.randn(1,1,20,20)
lenet = LeNet()
lenet(data)
"""
