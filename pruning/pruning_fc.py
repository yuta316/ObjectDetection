import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import sys

from matplotlib.ticker import ScalarFormatter

class Compressor(object):
  def __init__(self, model, cuda=False):
    #pruningするモデル
    self.model = model.cuda()
    self.num_layers = 0
    self.num_dropout_layers = 0
    self.dropout_rates = {}

    #層をカウントするメソッドを呼び出す
    self.count_layers()

    self.weight_masks = [None for _ in range(self.num_layers)]
    self.bias_masks = [None for _ in range(self.num_layers)]

    self.cuda = cuda

  def count_layers(self):
    for m in self.model.modules():
      #全結合層、Conv層を検索
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        self.num_layers += 1
      elif isinstance(m, nn.Dropout):
        self.dropout_rates[self.num_dropout_layers] = m.p
        self.num_dropout_layers += 1

  def prune(self):
    """
    ネットワーク全体でpruningする割合(%)を示す。
    """
    index = 0
    dropout_index = 0

    num_pruned, num_weights = 0, 0

    for m in self.model.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #重みの全データ数を取得
        num = torch.numel(m.weight.data)

        if type(m) == nn.Conv2d:
          if index == 0:
            alpha = 0.015
          else:
            alpha = 0.2
        else :
          if index == self.num_layers - 1:
            alpha = 0.25
          else:
            alpha = 1
        
        #byteTensorを使用してマスクを表し、乗算のためにfloatTensorに変換する
        #tordch.ge()で閾値を指定してマスクを作成する(閾値は重みの標準偏差)
        weight_mask  = torch.ge(m.weight.data.abs(), alpha*m.weight.data.std()).type('torch.FloatTensor')
        if self.cuda:
                    weight_mask = weight_mask.cuda()
        self.weight_masks[index] = weight_mask

        bias_mask = torch.ones(m.bias.data.size())
        if self.cuda:
          bias_mask = bias_mask.cuda()
        
        #conv2dレイヤー内のすべてのカーネルにおいて、いずれかのカーネルがすべて0の場合、バイアスを0に設定
        #線形レイヤーの場合、代わりにゼロ行を検索
        for i in range(bias_mask.size(0)): 
          if len(torch.nonzero(weight_mask[i]).size()) == 0:
            bias_mask[i] = 0
        self.bias_masks[index] = bias_mask

        index += 1

        layer_pruned = num - torch.nonzero(weight_mask).size(0)
        #logging.info('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
        bias_num = torch.numel(bias_mask)
        bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
        #logging.info('number pruned in bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

        num_pruned += layer_pruned
        num_weights += num

        #重みをかけることでpruning
        m.weight.data *= weight_mask
        m.bias.data *= bias_mask

      elif isinstance(m, nn.Dropout):
        #Dropout rate を更新
        mask = self.weight_masks[index-1]
        m.p = self.dropout_rates[dropout_index]*math.sqrt(torch.nonzero(mask).size(0) / torch.numel(mask))
        dropout_index += 1
        logging.info("new Dropout rate:", m.p)

    return num_pruned / num_weights
  
  def set_grad(self):
    index = 0
    for m in self.model.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.grad.data += self.weight_masks[index]
        m.bias.grad.data *= self.bias_masks[index]
        index+=1




model = モデル
#事前学習時みモデル
model.load_state_dict(torch.load("モデルまでのpath"))
model_comp = Compressor(model, cuda=False)

#pruning
model_comp.prune()


