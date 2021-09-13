import sys
import pathlib

currentdir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(currentdir)+"/../backbones/")

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

from ResNet import ResNet101

def obj_loc(score, threshold):
    """
    score: [width(height)]個分のピクセル信頼度
    """
    smax, sdis, sdim = 0, 0, score.size(0)
    minsize = int(math.ceil(sdim * 0.125))  #物体領域候補の最低サイズ0.125
    snorm = (score - threshold).sign()
    snormdiff = (snorm[1:] - snorm[:-1]).abs()

    szero = (snormdiff==2).nonzero()
    if len(szero)==0:
       zmin, zmax = int(math.ceil(sdim*0.125)), int(math.ceil(sdim*0.875))
       return zmin, zmax

    if szero[0] > 0:
       lzmin, lzmax = 0, szero[0].item()
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if szero[-1] < sdim:
       lzmin, lzmax = szero[-1].item(), sdim
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if len(szero) >= 2:
       for i in range(len(szero)-1):
           lzmin, lzmax = szero[i].item(), szero[i+1].item()
           lzdis = lzmax - lzmin
           lsmax, _ = score[lzmin:lzmax].max(0)
           if lsmax > smax:
              smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
           if lsmax == smax:
              if lzdis > sdis:
                 smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if zmax - zmin <= minsize:
        pad = minsize-(zmax-zmin)
        if zmin > int(math.ceil(pad/2.0)) and sdim - zmax > pad:
            zmin = zmin - int(math.ceil(pad/2.0)) + 1
            zmax = zmax + int(math.ceil(pad/2.0))
        if zmin < int(math.ceil(pad/2.0)):
            zmin = 0
            zmax =  minsize
        if sdim - zmax < int(math.ceil(pad/2.0)):
            zmin = sdim - minsize + 1
            zmax = sdim

    return zmin, zmax

class MCARResnet(nn.Module):
  """
  バックボーンにResNet101を利用する
  データセットはVOC2012(クラス数は20)
  """
  def __init__(self, model, topN, threshold,  num_classes=20,  vis=False):
    super(MCARResnet, self).__init__()
    self.features = nn.Sequential(
        model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4,
    )
    num_features = model.layer4[1].conv1.in_channels        #2048ch
    self.convclass = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
    self.num_classes = num_classes
    self.num_features = num_features
    self.topN = topN
    self.threshold = threshold
    self.vis = vis
    # image normalization
    self.image_normalization_mean = [0.485, 0.456, 0.406]
    self.image_normalization_std = [0.229, 0.224, 0.225]

  def forward(self, x):
    """
    1. グローバルストリーム : グローバル予測分布を取得する
    """

    """
    ① ResNetに画像を入力し、アクティベーションマップを取得する
        outputs : [b, 2048, 8, 8]
    """
    b, ch, height, width = x.size()
    global_acitivation = self.features(x)

    """
    ② グローバルプーリングでシングル(1 x 1)ベクトルにエンコード
        outputs : [b, 2048, 1, 1]
    """
    global_features = F.avg_pool2d(global_acitivation,  8, 8)

    """
    ③ 1×1畳み込みで予測スコアを取得する(チャネル数をクラス数に変換)
        outputs : [b, 20, 1, 1]
    """
    global_features = self.convclass(global_features)
    
    """
    ④ グローバル予測分布 シグモイドで[0, 1]に正規化
        outputs : [b, 20]
    """
    # シグモイドで[0, 1]に正規化
    global_stream = torch.sigmoid(global_features)
    global_stream = global_stream.view(global_stream.size(0), -1)

    """
    2. グローバル->ローカリゼーション
    2.1. アテンションマップの作成
    """
    """
    ① ResNetのアクティベーションマップに1x1Convをかけ、クラス固有のアクティベーションマップを得る。
        outputs : [b, 20, 8, 8]
    """
    camscore = self.convclass(global_acitivation.detach())

    """
    2.2. ローカルリージョンローカリゼーション
    """
    """
    ② [0,1] で 正規化
        outputs : [b, 20, 8, 8]
    """
    camscore = torch.sigmoid(camscore) 

    """
    ③ 入力サイズにアップサンプリングして入力画像のとの空間セマンティクスを調整する。
        [b, num_calsses, h, w]になり、各(h,w)でカテゴリに属する確率を表す。
        outputs : [b, 20, 64, 64]
    """
    camscore = F.interpolate(camscore, size=(height, width), mode='bilinear', align_corners=True)
 
    """
    ④ オブジェクト確率マップをx ,y軸に分解する
        outputs : [b, 20, 64]
    """
    wscore = F.max_pool2d(camscore, (height, 1)).squeeze(dim=2)
    hscore = F.max_pool2d(camscore, (1, width)).squeeze(dim=3)

    linputs = torch.zeros([b, self.topN, 3, height, width]).cuda()         # 領域候補の準備(torch.Size([b, 4, 3, 64, 64]))

    if self.vis:
      region_bboxs = toorch.FloatTensor(b, self.topN, 6)

    for i in range(b): # バッチ数でループ
      """
      2.3. 注意マップの選択
            カテゴリ信頼度で降順にソートし、上位topK個分の領域を用いる。
            linputsには各画像(バッチ)ごとtopK分の領域候補がアップサンプリングされ、格納される。
      """
      global_stream_inv, global_stream_ind = global_stream[i].sort(descending=True)
      # [20], [20]

      for j in range(self.topN):
        # 各カテゴリごとに取り出す
        xs = wscore[ i , global_stream_ind[ j ], : ].squeeze() # [64]
        ys = hscore[ i , global_stream_ind[ j ], : ].squeeze() # [64]
        if xs.max() == xs.min():
          xs = xs/xs.max()
        else:
          xs = (xs - xs.min())/(xs.max()-xs.min())
        if ys.max() == ys.min():
          ys = ys/ys.max()
        else:
          ys = (ys-ys.min())/(ys.max()-ys.min())

        x1, x2 = obj_loc(xs, self.threshold)
        y1, y2 = obj_loc(ys, self.threshold)

        linputs[ i : i+1, j ] = F.interpolate(x[ i : i+1, : , y1 : y2, x1 : x2], size=(height, width), mode='bilinear', align_corners=True)
        if self.vis == True:
          region_bboxs[i,j] = torch.Tensor([x1, x2, y1, y2, global_stream_ind[j].item(), global_stream[i, global_stream_ind[j]].item()])
    
    """
    Lobal Image Stream : ローカル領域を取得する
    """
    """
    ⓪ とりあえずバッチ関係なく候補をまとめる
         outputs: [bxtopK,3, 64, 64]
    """
    linputs = linputs.view( b*self.topN, 3, height, width )

    """
    ① 再度ResNetに入力、グローバルプーリング, 1x1、シグモイドによる正規化
         outputs: [bxtopK,2048, 8, 8] -> [bxtopk, 2048, 1, 1] -> [bxtopk, 20, 1, 1]
    """
    local_acitivation = self.features(linputs.detach())
    local_features = F.avg_pool2d(local_acitivation,  8, 8)
    local_features = self.convclass(local_features)
    local_stream = torch.sigmoid(local_features)

    """
    ② グローバルプーリングでカテゴリ信頼度を集約化
         outputs: [b, 20]
    """
    local_stream = F.max_pool2d(local_stream.reshape(b, self.topN, self.num_classes, 1).permute(0,3,1,2), (self.topN, 1))
    local_stream = local_stream.view(local_stream.size(0), -1)

    if self.vis == True:
      return global_stream, local_stream, region_bboxs
    else:
      return global_stream, local_stream, camscore, wscore, hscore, linputs

  def get_config_optim(self, lr, lrp):
    return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.convclass.parameters(), 'lr': lr},
    ]

def main():
    res101 = ResNet101()
    model = MCARResnet(model=res101, topN=4, threshold=0.5)

if __name__ == "__main__":
    main()