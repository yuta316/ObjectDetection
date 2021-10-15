class BackBone(nn.Module):
  def __init__(self, DownSample, num_classes):
    super(BackBone, self).__init__()
    self.in_ch = 32
    self.relu = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    self.avgpool = nn.AvgPool2d(8)

    # DarkNetを流用した特徴量抽出
    self.cbr = CBR(3, 32)
    self.layer1 = self._make_layer(DownSample,     64, 2, stride=2)
    self.layer2 = self._make_layer(DownSample,   128, 2, stride=2)

    # 特徴マップに対するアテンション
    self.attention_layer     = self._make_layer(DownSample,   256, 4, stride=1)
    self.attention_bn        = nn.BatchNorm2d(256)
    self.attention_conv     = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, bias=False)
    self.attention_bn_2     = nn.BatchNorm2d(num_classes)
    self.attention_conv_2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
    self.attention_conv_3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
    self.attention_bn_3     = nn.BatchNorm2d(1)
    self.attention_gap       = nn.AvgPool2d(64)

    # クラス分類
    self.in_ch = 128
    self.layer3 = self._make_layer(DownSample,   256, 8, stride=2)
    self.layer4 = self._make_layer(DownSample,   512, 8, stride=2)
    self.layer5 = self._make_layer(DownSample, 1024, 4, stride=2)
    self.class_conv   = nn.Conv2d(1792, num_classes, kernel_size=1, padding=0, bias=False)

    self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)

    self.midlayer1 = CBR(768, 256, 1, 1, 0)
    self.midlayer2 = CBR(256, 512, 3, 1, 1)
    self.midlayer3 = CBR(512, 256, 1, 1, 0)
    self.midlayer4 = CBR(256, 512, 3, 1, 1)
    self.midlayer5 = CBR(512, 256, 1, 1, 0)
    self.midlayer6 = CBR(256, 512, 3, 1, 1)
    self.midlayer  = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)

    self.botlayer1 = CBR(384, 128, 1, 1, 0)
    self.botlayer2 = CBR(128, 256, 3, 1, 1)
    self.botlayer3 = CBR(256, 128, 1, 1, 0)
    self.botlayer4 = CBR(128, 256, 3, 1, 1)
    self.botlayer5 = CBR(256, 128, 1, 1, 0)
    self.botlayer6 = CBR(128, 256, 3, 1, 1)
    self.botlayer  = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

    self.heatmaplayer1 = CBR(1792, 512, 1, 1, 0)
    self.heatmaplayer2 = CBR(20, 64, 3, 1, 1)
    self.heatmaplayer3 = nn.Conv2d(64, 20, kernel_size=1, stride=1, padding=0)
    self.topmaplayer1 = CBR(512, 512)
    self.topmaplayer2 = nn.Conv2d(512, 20, kernel_size=1, stride=1, padding=0)
    self.btmmaplayer1 = CBR(512, 512)
    self.btmmaplayer2 = nn.Conv2d(512, 20, kernel_size=1, stride=1, padding=0)
    self.lftmaplayer1 = CBR(512, 512)
    self.lftmaplayer2 = nn.Conv2d(512, 20, kernel_size=1, stride=1, padding=0)
    self.rgtmaplayer1 = CBR(512, 512)
    self.rgtmaplayer2 = nn.Conv2d(512, 20, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    x0 = self.cbr(x)     # [bn,   32,    416,   416]
    x1 = self.layer1(x0) # [bn,   64,    208,   208]
    x2 = self.layer2(x1) # [bn,  128,    104,   104]

    # アテンション
    attention = self.attention_bn(self.attention_layer(x2)) # [bn, 1024, x1/16, x1/16]
    attention = self.relu(self.attention_bn_2(self.attention_conv(attention))) # [bn, 20, x1/16, x1/16]
    self.attention = self.sigmoid(self.attention_bn_3(self.attention_conv_3(attention))) # [bn, 1, x1/16, x1/16]
    attention = self.attention_conv_2(attention) # [bn, 20, x1/16, x1/16]
    attention = self.attention_gap(attention) # [bn, 20, 1, 1]
    attention = self.sigmoid(attention.view(attention.size(0), -1)) # [bn, 20]

    # 特徴マップと特徴マップアテンションを掛け合わせる
    ax = x2 * self.attention
    x3 = self.layer3(ax) # [bn,   256,   52,   52]
    x4 = self.layer4(x3) # [bn,   512,   26,   26]
    x5 = self.layer5(x4) # [bn,  1024,   13,   13]

    p5 = self.toplayer(x5)          # [bn,  256,   13,   13]
    p4 = self._upsample_add(p5, x4) # [bn,  768,   26,   26]
    x4 = self.midlayer6(self.midlayer5(self.midlayer4(self.midlayer3(self.midlayer2(self.midlayer1(p4))))))
    # [bn, 512, 26, 26]
    p4 = self.midlayer(x4)

    p3 = self._upsample_add(p4, x3) # [bn, 384, 52, 52]

    x3 = self.botlayer6(self.botlayer5(self.botlayer4(self.botlayer3(self.botlayer2(self.botlayer1(p3))))))
    # [bn, 256, 52, 52]
    # p3 = self._upsample_add(p4, x3)
    # p2 = self._upsample_add(p3, ax)
    # # p4 = self.smooth1(p4)
    # # p3 = self.smooth2(p3)
    # # p2 = self.smooth3(p2)

    # classfy = torch.cat([F.avg_pool2d(x5, x5.shape[2]) ,F.avg_pool2d(x4, x4.shape[2]),F.avg_pool2d(x3, x3.shape[2])], dim=1)
    # classfy = self.class_conv(classfy)
    # classfy = self.sigmoid(classfy.view(classfy.size(0), -1))

    heatmap = torch.cat([
      F.upsample(x5, size=(x5.shape[2]*4, x5.shape[2]*4)),
      F.upsample(x4, size=(x4.shape[2]*2, x4.shape[2]*2)),
      x3], dim=1) # [bn, 1792, 52, 52]
    heatmap =  self.heatmaplayer1(heatmap) # [bn, 512, 52, 52]
    # heatmap =  self.heatmaplayer2(heatmap) # [bn, 20, 52, 52]

    topmap = self.topmaplayer2(self.topmaplayer1(heatmap)) # [bn, 20, 52, 52]
    btmmap = self.btmmaplayer2(self.btmmaplayer1(heatmap)) # [bn, 20, 52, 52]
    rgtmap = self.rgtmaplayer2(self.rgtmaplayer1(heatmap)) # [bn, 20, 52, 52]
    lftmap = self.lftmaplayer2(self.lftmaplayer1(heatmap)) # [bn, 20, 52, 52]

    # heatmap = self.sigmoid(F.upsample(heatmap, size=(heatmap.shape[2]*2, heatmap.shape[2]*2))) # [bn, 20, 104, 104]
    topmap = F.upsample(topmap, size=(topmap.shape[2]*2, topmap.shape[2]*2))
    btmmap = F.upsample(btmmap, size=(btmmap.shape[2]*2, btmmap.shape[2]*2))
    rgtmap = F.upsample(rgtmap, size=(rgtmap.shape[2]*2, rgtmap.shape[2]*2))
    lftmap = F.upsample(lftmap, size=(lftmap.shape[2]*2, lftmap.shape[2]*2))

    heatmap = topmap + btmmap + lftmap + rgtmap
    heatmap = self.heatmaplayer3(self.heatmaplayer2(heatmap))
    classfy = F.avg_pool2d(heatmap, heatmap.shape[2]) # [bn, 20, 1, 1]
    classfy = self.sigmoid(classfy.view(classfy.size(0), -1))

    return attention, classfy, topmap, btmmap, lftmap, rgtmap

  def _upsample_add(self, x, y):
    _, _, height, width = y.size()
    return torch.cat([F.upsample(x, size=(height, width), mode='bilinear') , y], dim=1)

  def _make_layer(self, block, ch, num_blocks, stride):
    # 初めのブロック以外は全てストライドが1
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_ch, ch, stride))
      self.in_ch = ch
    return nn.Sequential(*layers)