import torch.nn as nn
import torch

class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.l1 = nn.Linear(28*28, 300)
        self.l2 = nn.Linear(300, 100)
        self.l3 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        return self.l3(self.l2(self.l1(x)))
    
    def num_flat_features(self, x):
        #バッチサイズ除く
        size = x.size()[1:]
        #全ての次元を掛け合わせる
        num_features = 1
        for s in size:
            num_features*=s
        return num_features

#data = torch.randn(1,28,28)
#lenet = LeNet_300_100()
#lenet(data)

"""
[
Linear(in_features=784, out_features=300, bias=True), 
Linear(in_features=300, out_features=100, bias=True), 
Linear(in_features=100, out_features=10, bias=True)
]
"""