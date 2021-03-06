import math 
import torch
from torch import nn

#活性化関数にswishを採用
class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x*torch.sigmoid(x)

#Squeeze Excitaiton module
class SqueezeExcitation(nn.Module):
    """
    チャンネル(空間)方向のアテンション.
    SENetは畳み込み層の各チャネルを均等に出力せず適当に重みをかける.
    """
    def __init__(self, in_channel, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        
        self.se = nn.Sequential(
            #チャネルごとの代表値として画素値平均をとる(1x1xC)
            nn.AdaptiveAvgPool2d(1),
            #チャネル数削減(1x1xC/r)
            nn.Conv2d(in_channel, reduced_dim, kernel_size=(1,1)),
            Swish(),
            #チャネル数復元(1x1xC)
            nn.Conv2d(reduced_dim, in_channel, kernel_size=(1,1)),
            #正規化
            nn.Sigmoid()
            )
        
    def forward(self, x):
        #各チャンネルに重みづけ
        return x*self.se(x)

#ConvBNReluレイヤ
class ConvBNRelu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNRelu, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=0,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            Swish(),
        )
       
    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p//2, p-p//2, p//2, p-p//2]

#MBConvBlockレイヤ
class MBConvBlock(nn.Module):
    def __init__(self,
                     in_channel,
                     out_channel, 
                     expand_ratio, 
                     stride, 
                     kernel_size,
                     reduction_ratio=4,
                     drop_connetct_rate=0.2):
        super(MBConvBlock, self).__init__()
        
        self.drop_connetct_rate = drop_connetct_rate
        self.use_residual = (in_channel==out_channel) & (stride==1)
        
        assert stride in [1,2]
        assert kernel_size in [3,5]
        
        hidden_dim = int(in_channel * expand_ratio)
        reduced_dim = max(1, int(in_channel / reduction_ratio))
        
        #ネットワーク
        layers = []
        if in_channel != hidden_dim:
            layers += [ConvBNRelu(in_channel, hidden_dim,1)]
        
        layers += [
            #depth-wise
            ConvBNRelu(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            #se
            SqueezeExcitation(hidden_dim, reduced_dim),
            #pixel-wise
            nn.Conv2d(hidden_dim, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        ]
        
        self.conv = nn.Sequential(*layers)
                       
    def _drop_connetct(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connetct_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob)*binary_tensor
                       
    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connetct(self.conv(x))
        else:
            return self.conv(x)


params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}

class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, 
                dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()
    
        # expand_ratio, channel, repeats, stride, kernel_size                   
        settings = [
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112                   
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56                   
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28                   
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14                   
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14                   
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7                   
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7]                  
        ]
        
        out_channel = _round_filters(32,width_mult)
        features = [ConvBNRelu(3, out_channel, 3, stride=2)]
        
        in_channel = out_channel
        
        for t , c, n, s, k in settings:
            out_channel = _round_filters(c,width_mult)
            repeats = int(math.ceil( n * depth_mult))
            for i in range(repeats):
                stride = s if i==0 else 1
                features += [MBConvBlock(in_channel, out_channel, t, stride, k)]
                in_channel = out_channel
                
        last_chanel = int(math.ceil(1280*width_mult))
        features += [ConvBNRelu(in_channel, last_chanel, 1)]
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_chanel, num_classes),
        )
        
        #重み初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2,3])
        x = self.classifier(x)
        return x

try: 
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def _efficientnet(arch, pretraind, progress, **kwargs):
    width_mult, depth_mult, _, drop_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, drop_rate, **kwargs)
    
    if pretraind:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        
        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']
        model.load_state_dict(state_dict, strict=False)
    
    return model

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)
def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs, num_classes=10)
def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b4', pretrained, progress, **kwargs)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))

