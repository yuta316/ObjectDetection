import torch.nn as nn
import math
import torch

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    
    new_v = max(min_value, int(v + divisor/2)//divisor*divisor)
    if new_v < 0.9*v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        return self.relu(x+3)/6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
        
    def forward(self, x):
        return x*self.sigmoid(x)

def C3BR(in_channel, out_channel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size =3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channel),
        h_swish()
    )

def C1BR(in_channel, out_channel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channel),
        h_swish()
    )

class SElayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y

class InvertedResidual(nn.Module):
    def __init__(self,inp, hidden_dim, outp, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1,2]

        #恒等写像か否かのバイナリ
        self.identity = stride == 1 and inp == outp
        
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                SElayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, outp, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outp),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SElayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, outp, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outp),
            )
        
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self,cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        
        #初めのレイヤ
        in_channel = _make_divisible(16* width_mult,8)
        layers = [C3BR(3,in_channel, stride=2)]
        
        #inverted residual blocks
        block = InvertedResidual
       
        for kernel_size, expand_ratio, channel, use_se, use_hs, stride in self.cfgs:
            out_channel = _make_divisible(channel * width_mult, 8)
            exp_size = _make_divisible(in_channel*expand_ratio, 8)
            layers.append(block(in_channel, exp_size,out_channel,kernel_size=kernel_size, stride=stride,use_se=use_se, use_hs=use_hs))
            in_channel = out_channel
       
        self.features = nn.Sequential(*layers)
        
        #最後のレイヤ
        self.conv = C1BR(in_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        out_channel = {'large': 1280, 'small': 1024}
        out_channel = _make_divisible(out_channel[mode]*width_mult, 8) if width_mult > 1.0 else out_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, out_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(out_channel, num_classes),
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, mode='small', **kwargs)

#model = mobilenetv3_large()
"""
MobileNetV3(
  (features): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): h_swish(
        (sigmoid): h_sigmoid(
          (relu): ReLU6(inplace=True)
        )
      )
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Identity()
        (4): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
        (4): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
        (4): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=72, out_features=24, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=24, out_features=72, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        (4): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=120, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=120, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        (4): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=120, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=120, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        (4): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=200, bias=False)
        (4): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(200, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        (4): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        (4): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        (4): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=480, out_features=120, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=120, out_features=480, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
        (4): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=672, out_features=168, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=168, out_features=672, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
        (4): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=672, out_features=168, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=168, out_features=672, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=960, out_features=240, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=240, out_features=960, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SElayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=960, out_features=240, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=240, out_features=960, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (conv): Sequential(
    (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): h_swish(
      (sigmoid): h_sigmoid(
        (relu): ReLU6(inplace=True)
      )
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (classifier): Sequential(
    (0): Linear(in_features=960, out_features=1280, bias=True)
    (1): h_swish(
      (sigmoid): h_sigmoid(
        (relu): ReLU6(inplace=True)
      )
    )
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=1280, out_features=1000, bias=True)
  )
)
"""