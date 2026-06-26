import pdb
import torch
from torch import nn


class HookTool:
    def __init__(self):
        self.grad_block = {}
        self.fmap_block = {}

    def backward_hook(self, module, grad_input, grad_output):
        self.grad_block['grad_input'] = grad_input
        self.grad_block['grad_output'] = grad_output

    def forward_hook(self, module, fmap_input, fmap_output):
        self.fmap_block['fmap_input'] = fmap_input
        self.fmap_block['fmap_output'] = fmap_output

    def get_grad(self):
        return self.grad_block

    def get_fmap(self):
        return self.fmap_block


'''Build ResNet18'''
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers, zero_init_residual=True):
        """
        layers: list, number of blocks every layer, eg: ResNet18 -> [2, 2, 2, 2]
        """
        super(ResNet, self).__init__()
        self.inplanes = 64  # inplanes: int, number of channels in the input data
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1])
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2])
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1)

        # 模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # 层次化设计
    def _make_layer(self, block, planes, blocks):
        """
        block: BasicBlock
        planes: int, the number of channels in the input data
        blocks: int, the number of blocks
        """
        stride = 1  # 固定为1
        downsample = None  # 是否采用downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # 使用layers存储每个layer
        layers = []
        layers.append(block(self.inplanes, planes, downsample))  # 保证通道数一致
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [-1, 1, 256, 25]
        x = self.conv1(x)  # [-1, 64, 128, 13]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [-1, 64, 64, 7]

        x = self.layer1(x)  # [-1, 64, 64, 7]
        x = self.layer2(x)  # [-1, 128, 64, 7]
        x = self.layer3(x)  # [-1, 256, 64, 7]
        x = self.layer4(x)  # [-1, 512, 64, 7]

        x = self.avgpool(x)  # [-1, 512, 1, 1]
        x = torch.flatten(x, 1)  # [-1, 512]
        x = self.fc(x)  # [-1, 1]

        return x


'''Build AutoEncoder'''
class DoubleConv(nn.Module):
    """conv -> BN -> relu -> dropout -> conv -> BN -> relu"""
    def __init__(self, in_channels, out_channels, dropout_rate, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.double_conv(inputs)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, inputs):
        return self.down(inputs)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(Up, self).__init__()
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, inputs):
        return self.upsample_conv(inputs)


class Encoder(nn.Module):
    def __init__(self, dropout_rate):
        super(Encoder, self).__init__()
        # initialize DoubleConv, Down and Linear blocks
        self.convblock = DoubleConv(in_channels=1, out_channels=64, dropout_rate=dropout_rate)
        self.down1 = Down(in_channels=64, out_channels=128, dropout_rate=dropout_rate + 0.1)
        self.down2 = Down(in_channels=128, out_channels=256, dropout_rate=dropout_rate + 0.2)
        # initialize other layers
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, inputs):
        # inputs: [-1, 1, 100, 4]

        x = self.convblock(inputs)  # [-1, 64, 100, 4]
        x = self.down1(x)  # [-1, 128, 50, 2]
        vector = self.down2(x)  # [-1, 256, 25, 1]

        return vector


class PredHead(nn.Module):
    def __init__(self):
        super(PredHead, self).__init__()
        self.resnet18 = ResNet(layers=[2, 2, 2, 2])

    def forward(self, x):
        inputs = torch.reshape(x, (-1, 1, 256, 25))
        mrl = self.resnet18(inputs)
        return mrl


class Predictor(nn.Module):
    def __init__(self, dropout_rate):
        super(Predictor, self).__init__()
        hooktool = HookTool()
        self.grad_block = hooktool.get_grad()

        self.encoder = Encoder(dropout_rate)
        self.predhead = PredHead()
        self.predhead.register_full_backward_hook(hooktool.backward_hook)

    def forward(self, x):
        x = self.encoder(x)
        outputs = self.predhead(x)
        return outputs

    def getencoder(self):
        return self.encoder

    def getpredhead(self):
        return self.predhead


class Decoder(nn.Module):
    def __init__(self, dropout_rate):
        super(Decoder, self).__init__()
        # initialize upsample and output blocks
        self.up1 = Up(in_channels=257, out_channels=128, dropout_rate=dropout_rate + 0.1)
        self.up2 = Up(in_channels=128, out_channels=64, dropout_rate=dropout_rate)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), padding=0),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.up1(inputs)
        x = self.up2(x)

        #prepare decoder's output
        x = self.output(x)
        outputs = torch.reshape(x, (-1, 1, 100, 4))

        return outputs


class Autoencoder(nn.Module):
    def __init__(self, dropout_rate, encoder=None):
        super(Autoencoder, self).__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Encoder(dropout_rate)
        self.decoder = Decoder(dropout_rate)

    def forward(self, inputs):
        latent_vector = self.encoder(inputs)
        seqs_pred = self.decoder(latent_vector)

        return seqs_pred

    def getencoder(self):
        return self.encoder

    def getdecoder(self):
        return self.decoder


def get_encoder_parameters(model):
    encoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
    return encoder_params


def get_decoder_parameters(model):
    decoder_params = []
    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_params.append(param)
    return decoder_params