import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps, ImageFilter

import torch
from torchvision import transforms
import torch.nn as nn
import torchvision
import time
import torch.nn.functional as F

from torch.autograd import Variable
import random

import math

class BasicBlock(nn.Module):
  # チャンネルを何倍に増やして出力を返すか
  expansion = 1

  def __init__(self, in_ch, ch, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=3, stride=striide, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(ch)
    self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(ch)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_ch != self.expansion*ch:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_ch, self.expansion*ch, kernel_size=1, stride=striide, bias=False),
          nn.BatchNorm2d(self.expansion*ch)
      ) 
    
  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class BottleNeck(nn.Module):
  expansion = 4

  def __init__(self, in_ch, ch, stride=1):
    super(BottleNeck, self).__init__()
    self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(ch)
    self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(ch)
    self.conv3 = nn.Conv2d(ch, self.expansion*ch, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*ch)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_ch != self.expansion*ch:
      self.shortcut = nn.Sequential(
           nn.Conv2d(in_ch, self.expansion*ch, kernel_size=1, stride=stride, bias=False),
           nn.BatchNorm2d(self.expansion*ch)
      )
  
  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=20):
    super(ResNet, self).__init__()
    self.in_ch = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear1 = nn.Linear(512*block.expansion*16, 4096)
    self.linear2 = nn.Linear(4096, 4096)
    self.linear3 = nn.Linear(4096, 20)

  def _make_layer(self, block, ch, num_blocks, stride):
    # 初めのブロック以外は全てストライドが1
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_ch, ch, stride))
      self.in_ch = ch*block.expansion
    return nn.Sequential(*layers)
  
  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = F.relu(self.linear1(out))
    out = F.relu(self.linear2(out))
    out = self.linear3(out)
    return out

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])