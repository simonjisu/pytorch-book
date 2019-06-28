import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, config):
        """
        config: 리스트, 리스트 안에는 튜플의 형태
        > convoltion 연산은 (커널개수, 커널크기, 스트라이드, 패딩) 순서
        > pooling 연산은 커널크기(=스트라이드 같게 설정)를 넣는다. 
         - example :
         config = [('ch_in', 3), ('n_in', 32),
                   ('conv1', (32, 7, 1, 1)), ('pool1', 2), 
                   ('conv2', (64, 5, 1, 1)), ('pool2', 2), 
                   ('conv3', (128, 3, 1, 0)), ('pool3', 2),
                   ('fc', (250, 50, 10))]
        """
        super(CNN, self).__init__()
        check_config = lambda x: True if dict(config).get(x) is not None else False
        for x in ["ch_in", "n_in", "fc"]:
            assert check_config(x), "must insert '{}' in config".format(x)
        
        self.flatten = lambda x: x.view(x.size(0), -1)
        self.convs = nn.Sequential(*self._make_layers(config, typ="convs"))
        self.fc = nn.Sequential(*self._make_layers(config, typ="fc"))
    
    def forward(self, x):
        # Convolutional Layer
        x = self.convs(x) 
        # flatten vectors
        x = self.flatten(x)
        # Fully Connect Layer
        x = self.fc(x)
        return x

    def _make_layers(self, config, typ):
        ch_in = dict(config)["ch_in"]
        layers = []
        if typ == "convs":
            for (k, v) in config:
                if "conv" in k:
                    ch_out, k, s, p = v
                    layers += [nn.Conv2d(in_channels=ch_in, 
                                         out_channels=ch_out, 
                                         kernel_size=k, 
                                         stride=s, 
                                         padding=p),
                               nn.BatchNorm2d(ch_out),
                               nn.ReLU(inplace=True)]
                    ch_in = ch_out
                elif "pool" in k:
                    layers += [nn.MaxPool2d(kernel_size=v, stride=v)]
        elif typ == "fc":
            features = [self._get_fc_input(config, dict(config)["n_in"])] + list(dict(config)["fc"])
            for i, (f_in, f_out) in enumerate(zip(features[:-1], features[1:])):
                if i == (len(features) - 2):
                    layers += [nn.Linear(f_in, f_out)]
                else:
                    layers += [nn.Linear(f_in, f_out), 
                               nn.ReLU(inplace=True)]
        else:
            assert False, "insert correct 'typ' in _make_layers "
            
        return layers
    
    def _get_fc_input(self, config, n_in):
        """
        n_in: size of image(height or weight, shuold be same)
        """
        assert isinstance(config, list), "config must be list type"
        config = dict(config)
        n_blocks = sum([True for k in config.keys() if 'conv' in k]) 
        for i in range(1, n_blocks+1):
            ch_out, k, s, p = config.get("conv{}".format(i))
            pool_k = config.get("pool{}".format(i))

            conv_n_out = (n_in + 2*p - k)/s + 1
            pool_n_out = (conv_n_out - pool_k)/pool_k + 1
            n_in = pool_n_out

        fc_input = int(ch_out*n_in*n_in)
        return fc_input
        
