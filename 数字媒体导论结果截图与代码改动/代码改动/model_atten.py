import torch
import torch.nn as nn
from torchvision.models import vgg19, resnet50,  VGG19_Weights,resnet18
import torch.nn.functional as F
from .layers import adain, NonLocalBlock

#定义自注意力子层
class SelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)

    def forward(self, x):
        # 假设输入 x 的形状为 (seq_len, batch_size, in_dim)
        # 注意：这里 seq_len 表示序列长度，对于图像可以理解为空间维度的大小
        # 使用自注意力层
        x, _ = self.self_attention(x, x, x)

        return x

class VGG(nn.Module):   # output relu4-1
    def __init__(self):
        super(VGG, self).__init__()
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.model = nn.Sequential(*model.features[:21])
        self._freeze_params()

    def forward(self, x):
        x = self.model(x)
        return x

    def _freeze_params(self):
        for p in self.model.parameters():
            p.requires_grad = False


class SAB(nn.Module):
    def __init__(self, identity=False):
        super(SAB, self).__init__()
        model = resnet50()
        self.model = nn.Module()
        self.model.conv1 = model.conv1
        self.model.bn1 = model.bn1
        self.model.relu = model.relu
        self.model.maxpool = model.maxpool

        self.model.layer1 = model.layer1
        self.model.layer2 = model.layer2

        self.identity = identity

        self.vgg = VGG()

        # 添加自注意力子层，嵌入向量维度为512，设置8个注意力头
        self.self_attention = SelfAttentionLayer(in_dim=512, num_heads=8)

        self._init_weights()

    def forward(self, x):
        # aligned to the output of VGG
        sty = self.vgg(x)
        aes = self.model.conv1(x)
        aes = self.model.bn1(aes)
        aes = self.model.relu(aes)
        aes = self.model.maxpool(aes)

        aes = self.model.layer1(aes)
        aes = self.model.layer2(aes)

        output = adain(aes, sty)
        #此处output的维度为[64,512,28,28]，开始将addin后的特征传入注意力子层
        bs = output.size(0)
        #进行维度变换以匹配注意力子层的输入
        output = output.view( 28 * 28,bs, 512)
        #自注意力计算
        output = self.self_attention(output)
        #将维度变回去
        output = output.view(bs, 512, 28, 28)

        if self.identity:
            output += aes

        return F.relu(output)

    def _init_weights(self):
    
        self.model.load_state_dict(torch.load('/home/liukai/likeyao/BAID/checkpoint/ResNet_Pretrain/epoch_39.pth',map_location='cpu'),
                                    strict=False)


class GAB(nn.Module):
    def __init__(self):
        super(GAB, self).__init__()
        model = resnet50()
        self.model = nn.Module()
        self.model.conv1 = model.conv1
        self.model.bn1 = model.bn1
        self.model.relu = model.relu
        self.model.maxpool = model.maxpool

        self.model.layer1 = model.layer1
        self.model.layer2 = model.layer2
        self.model.layer3 = model.layer3

        # 添加自注意力子层，嵌入向量维度为1024，设置8个注意力头
        self.self_attention = SelfAttentionLayer(in_dim=1024, num_heads=8)

        self._init_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        #x的维度为[64,1024,14,14]
        bs = x.size(0)
        #进行维度变换以匹配注意力子层的输入
        x = x.view( 14 * 14,bs, 1024)
        # 使用自注意力子层
        x = self.self_attention(x)
        #将维度变回去
        x = x.view( bs, 1024, 14, 14)
        return x

    def _init_weights(self):
        #device = next(self.model.parameters()).device

        # 打印当前设备
        #print("Model is currently on device:", device)
        self.model.load_state_dict(torch.load('/home/liukai/likeyao/BAID/checkpoint/ResNet_Pretrain/epoch_39.pth',map_location='cpu'),
                                    strict=False)


class SAAN_ATTEN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.GenAes = GAB()
        self.StyAes = SAB()

        self.NLB = NonLocalBlock(in_channels=1536)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))

        self.bn = nn.BatchNorm2d(num_features=1536)

        self.predictor = nn.Sequential(
            nn.Linear(1536 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
            nn.Sigmoid(),
        )

        self._initial_weights()

    def forward(self, x):
        gen_aes = self.GenAes(x)
        sty_aes = self.StyAes(x)

        sty_aes = self.max_pool(sty_aes)

        all_aes = torch.cat((sty_aes, gen_aes), 1)
        all_aes = self.NLB(all_aes)

        all_aes = self.avg_pool(all_aes)
        all_aes = self.bn(all_aes)

        fc_input = torch.flatten(all_aes, start_dim=1)

        output = self.predictor(fc_input)

        return output

    def _initial_weights(self):
        for m in self.bn.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)


class ResNetPretrain(nn.Module):
    def __init__(self, num_classes):
        super(ResNetPretrain, self).__init__()
        model = resnet50()
        self.model = model
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=num_classes, bias=True),
            nn.Softmax(dim=1),
        )

        self._initial_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)
        features_flat = self.model.avgpool(features)
        features_flat = torch.flatten(features_flat, 1)
        output = self.model.fc(features_flat)

        return features, output

    def _initial_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0.0)