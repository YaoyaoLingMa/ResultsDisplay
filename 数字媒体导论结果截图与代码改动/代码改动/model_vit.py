import torch
import torch.nn as nn
from torchvision.models import vgg19, resnet50,  VGG19_Weights,resnet18
import torch.nn.functional as F
from .layers import adain, NonLocalBlock
import timm
LAMDA = 0.3
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

        self._init_weights()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        return x

    def _init_weights(self):
        #device = next(self.model.parameters()).device

        # 打印当前设备
        #print("Model is currently on device:", device)
        self.model.load_state_dict(torch.load('/home/liukai/likeyao/BAID/checkpoint/ResNet_Pretrain/epoch_39.pth',map_location='cpu'),
                                    strict=False)


class SAAN_VIT(nn.Module):
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

        #定义预训练的ViT-base
        self.vision_transformer = timm.create_model('vit_base_patch16_224', pretrained=True)
        #定义ViT的特征提取器，去掉ViT的分类头
        self.vit_feature_extractor = torch.nn.Sequential(*list(self.vision_transformer.children())[:-2])
        #定义ViT的回归头
        self.vit_predictor = nn.Sequential(
            nn.Linear(768, 2048),
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

        #x的维度是[bs,3,224,224]，将x送入ViT的特征提取器来提取特征
        vit_feature = self.vit_feature_extractor(x)[:,0]
        #通过ViT的回归头得到ViT的输出
        vit_output =self.vit_predictor(vit_feature)
        #将SAAN的输出与ViT的输出按一定比例融合，比例通过LAMDA控制
        output = output*LAMDA + vit_output*(1.0-LAMDA)

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