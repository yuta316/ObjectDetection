import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import sys

def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='../data', 
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=1)
    testset = torchvision.datasets.MNIST(root='./drive/MyDrive/download/mnist', 
                                            train=False,
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=1)
    
    return trainloader, testloader