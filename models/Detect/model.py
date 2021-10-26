class CBR(nn.Module):
  # def __init__(self, block, depth, num_classes=20):
  def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
    super(CBR, self).__init__()

    self.conv=nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
    self.bn = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU(inplace=False)
  
  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))

class FeatureAttention(nn.Module):
  def __init__(self, ch, reduction=16):
    super(FeatureAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
        nn.Linear(ch, ch//reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(ch//reduction, ch, bias=False),
        nn.Sigmoid(),
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    feature_attention = self.avg_pool(x).view(b, c)
    feature_attention = self.fc(feature_attention).view(b, c, 1, 1)
    return x * feature_attention.expand_as(x)

class DownSample(nn.Module):
  expansion = 2

  def __init__(self, in_ch, out_ch, stride):
    super(DownSample, self).__init__()
    self.cbr1 = CBR(in_ch, out_ch, 3, stride, 1)
    self.cbr2 = CBR(out_ch, in_ch, 1, 1, 0)
    self.cbr3 = CBR(in_ch, out_ch, 3, 1, 1)
    self.feature_attention = FeatureAttention(out_ch)

    self.skip = nn.Sequential()
    if stride != 1 or in_ch != out_ch:
      self.skip = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 1, stride, 0),
          nn.BatchNorm2d(out_ch)
      )

  def forward(self, x):
    residual = x
    x = self.cbr1(x)
    x = self.cbr2(x)
    x = self.cbr3(x)
    x = self.feature_attention(x)
    residual = self.skip(residual)
    x = x + residual
    return x

class BackBone(nn.Module):
  def __init__(self, DownSample, num_classes):
    super(BackBone, self).__init__()
    self.in_ch = 32
    self.relu = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    self.avgpool = nn.AvgPool2d(8)

    self.cbr = CBR(3, 32)
    self.layer1 = self._make_layer(DownSample,     64, 2, stride=2)
    self.layer2 = self._make_layer(DownSample,   128, 2, stride=2)
    self.layer3 = self._make_layer(DownSample,   256, 8, stride=2)
    self.layer4 = self._make_layer(DownSample,   512, 8, stride=2)
    self.layer6 = self._make_layer(DownSample,    1024, 4, stride=2)
  

    self.in_ch = 1024
    self.reverse_layer1 = self._make_layer(DownSample,   512, 2, stride=1)
    self.in_ch = 1024
    self.reverse_layer2 = self._make_layer(DownSample,   512, 2, stride=1)
    self.in_ch = 768
    self.reverse_layer3 =  self._make_layer(DownSample,   512, 2, stride=1)

    self.in_ch = 704
    self.reverse_layer4 =  self._make_layer(DownSample,   128, 2, stride=1)
    self.in_ch = 704
    self.middle_layer = self._make_layer(DownSample,   128, 1, stride=1)

    self.in_ch = 128
    self.topmaplayer1 = CBR(128, 128)
    self.topmaplayer2 = self._make_layer(DownSample,   20, 2, stride=1)
    self.topmaplayer3 = nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)
    self.in_ch = 128
    self.btmmaplayer1 = CBR(128, 128)
    self.btmmaplayer2 = self._make_layer(DownSample,   20, 2, stride=1)
    self.btmmaplayer3 = nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)
    self.in_ch = 128
    self.lftmaplayer1 = CBR(128, 128)
    self.lftmaplayer2 = self._make_layer(DownSample,   20, 2, stride=1)
    self.lftmaplayer3 = nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)
    self.in_ch = 128
    self.rgtmaplayer1 = CBR(128, 128)
    self.rgtmaplayer2 = self._make_layer(DownSample,   20, 2, stride=1)
    self.rgtmaplayer3 = nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)
    self.in_ch = 128
    self.centermaplayer1 = CBR(128, 20)
    self.centermaplayer2 = self._make_layer(DownSample,   20, 2, stride=1)
    self.centermaplayer3 = nn.Conv2d(20, 20, kernel_size=1, stride=1, padding=0)

    # 特徴マップに対するアテンション
    self.in_ch = 128
    self.attention_layer     = self._make_layer(DownSample,   20, 1, stride=1)
    # self.attention_conv1    = CBR(128, 20, 1,1,0)
    self.attention_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1, bias=False)
    self.attention_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
    self.attention_bn     = nn.BatchNorm2d(1)
    self.attention_gap       = nn.AvgPool2d(80)
    
  def forward(self, x):
    x = self.cbr(x) # torch.Size([2, 64, 300, 300])
    x = self.layer1(x) # torch.Size([2, 64, 150, 150])
    x0 = self.layer2(x) # torch.Size([2, 128, 75, 75])
    x1 = self.layer3(x0) # torch.Size([2, 256, 38, 38])
    x2 = self.layer4(x1) # torch.Size([2, 512, 19, 19])
    x3 = self.layer6(x2) # torch.Size([2, 1024, 10, 10])

    p3 = self.reverse_layer1(x3) # torch.Size([2, 512, 10, 10])
    p2 = self._upsample_add(p3, x2) # torch.Size([2, 1024, 10, 10])
    p2 = self.reverse_layer2(p2) # torch.Size([2, 512, 10, 10])
    p1 = self._upsample_add(p2, x1) # torch.Size([2, 768, 20, 20])
    p1 = self.reverse_layer3(p1) # torch.Size([2, 512, 20, 20])
    p0 = self._upsample_add(p1,  torch.cat([x0, F.max_pool2d(x, 2)], dim=1)) # torch.Size([2, 704, 80, 80])

    feature_map = self.middle_layer(p0) # torch.Size([2, 128, 80, 80]) torch.Size([2, 64, 160, 160])
    attention = self.attention_layer(feature_map) # [bn, 128, 80, 80]
    self.attention = self.sigmoid(self.attention_bn(self.attention_conv3(attention))) # [bn, 1, 19, 19]
    attention = self.attention_conv2(attention) # [bn, 20, 80, 80]
    attention = F.avg_pool2d(attention,  attention.shape[2]) # [bn, 20, 1, 1]
    attention = self.sigmoid(attention.view(attention.size(0), -1)) # [bn, 20]

    feature_map = feature_map * self.attention
    topmap     = self.sigmoid(self.topmaplayer3(self.topmaplayer2(feature_map))) # [bn, 20, 38, 38]
    btmmap    = self.sigmoid(self.btmmaplayer3(self.btmmaplayer2(feature_map)) )# [bn, 20, 38, 38]
    lftmap       = self.sigmoid(self.lftmaplayer3(self.lftmaplayer2(feature_map)))# [bn, 20, 38, 38]
    rgtmap      = self.sigmoid(self.rgtmaplayer3(self.rgtmaplayer2(feature_map)) )# [bn, 20, 38, 38]
    centermap = self.sigmoid(self.centermaplayer3(self.centermaplayer2(feature_map))) # [bn, 20, 38, 38]
    
    classify=0
    # attention = 0

    return attention, classify, topmap, btmmap, lftmap, rgtmap, centermap

  def _make_layer(self, block, ch, num_blocks, stride):
    # 初めのブロック以外は全てストライドが1
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_ch, ch, stride))
      self.in_ch = ch
    return nn.Sequential(*layers)

  def _upsample_add(self, x, y):
    _, _, height, width = y.size()
    return torch.cat([F.upsample(x, size=(height, width), mode='bilinear') , y], dim=1)