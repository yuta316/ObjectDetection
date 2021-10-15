class PyramidPooling(nn.Module):
  def __init__(self, in_channels, pool_sizes):
    super(PyramidPooling, self).__init__()
    self.pool_len = len(pool_sizes)
    # 各解像度の出力チャネル数は等分する
    out_channels = int(in_channels / len(pool_sizes))

    for idx, pool_size in enumerate(pool_sizes):
      exec('self.avgpool_{} = nn.AdaptiveAvgPool2d(output_size={})'.format(idx+1, pool_size))
      exec('self.conv_{} = nn.Conv2d({}, {}, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)'.format(idx+1, in_channels, out_channels))
      exec('self.bn_{} = nn.BatchNorm2d({})'.format(idx+1, out_channels))
    self.relu = nn.ReLU(inplace = True)

  def forward(self, x):
    _, _, height, width = x.shape
    output = [x]
    for idx in range(1, self.pool_len+1):
      exec('out{} = self.relu(self.bn_{}(self.conv_{}(self.avgpool_{}(x))))'.format(idx, idx, idx, idx))
      exec("out{} = F.interpolate(out{}, size=(height, width), mode='bilinear', align_corners=True)".format(idx, idx))
      exec('output.append(out{})'.format(idx))
    output = torch.cat(output, dim=1)
    return output