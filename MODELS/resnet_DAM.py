# coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        #torch.manual_seed(1)
        #torch.cuda.manual_seed(1)
        self.inplanes = 64
        self.hidden_dim=64
        self.bidirect=True
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc0 = nn.Linear(512 * block.expansion, 256)
        self.fc1 = nn.Linear(256, num_classes)
        self.fc0_ = nn.Linear(512 * block.expansion, 256)
        self.fc1_ = nn.Linear(256, 2)
        self.lstm_row = nn.GRU(1, self.hidden_dim, bidirectional=self.bidirect, num_layers=2)
        self.lstm_col = nn.GRU(self.hidden_dim*2, self.hidden_dim, bidirectional=self.bidirect, num_layers=2)
        self.lstm_row2 = nn.GRU(self.hidden_dim*2, self.hidden_dim, bidirectional=self.bidirect, num_layers=2)
        self.lstm_col2 = nn.GRU(self.hidden_dim*2, self.hidden_dim, bidirectional=self.bidirect, num_layers=2)
        #self.lstm_col = nn.GRU(self.hidden_dim*2, self.hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)
        self.hidden2tag_1 = nn.Linear(self.hidden_dim * 2, 64)
        self.hidden2tag_3 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.hidden_row = torch.zeros(2*2, 1, self.hidden_dim).cuda()
        self.hidden_col = torch.zeros(2*2, 1, self.hidden_dim).cuda()
        self.hidden_row2 = torch.zeros(2*2, 1, self.hidden_dim).cuda()
        self.hidden_col2 = torch.zeros(2*2, 1, self.hidden_dim).cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        xc = self.fc0(x)
        xc2 = self.fc1(xc)
        xp = self.fc0_(x)
        xp2=self.fc1_(xp)
        xc2 = xc2.unsqueeze(0)
        input_row = xc2.view(1, -1, 1).permute(1, 0, 2).contiguous()
        #print(input_row.shape)
        lstm_R_out, self.hidden_row = self.lstm_row(input_row, self.hidden_row)
        #print(lstm_R_out.shape)
        lstm_R_out = lstm_R_out.view(-1, lstm_R_out.size(2))
        lstm_R_out = lstm_R_out.view(xc2.size(1), xc2.size(2), xc2.size(0), -1)
        #print(lstm_R_out.shape)

        input_col = lstm_R_out.permute(1, 0, 2, 3).contiguous()####
        input_col = input_col.view(-1, input_col.size(2), input_col.size(3)).contiguous()
        lstm_C_out, self.hidden_col = self.lstm_col(input_col, self.hidden_col)
        lstm_C_out = lstm_C_out.view(xc2.size(2), xc2.size(1), xc2.size(0), -1).permute(1, 0, 2, 3).contiguous()
        #print(lstm_C_out.shape)

        input_row2 = lstm_C_out.view(-1, lstm_C_out.size(2), lstm_C_out.size(3)).contiguous()
        #print(input_row2.shape)
        lstm_R_out2, self.hidden_row2 = self.lstm_row2(input_row2, self.hidden_row2)
        #print(lstm_R_out2.shape)
        lstm_R_out2 = lstm_R_out2.view(-1, lstm_R_out2.size(2))
        lstm_R_out2 = lstm_R_out2.view(xc2.size(1), xc2.size(2), xc2.size(0), -1)
        #print(lstm_R_out2.shape)

        input_col2 = lstm_R_out2.permute(1, 0, 2, 3).contiguous()####
        input_col2 = input_col2.view(-1, input_col2.size(2), input_col2.size(3)).contiguous()
        lstm_C_out2, self.hidden_col2 = self.lstm_col2(input_col2, self.hidden_col2)
        lstm_C_out2 = lstm_C_out2.view(xc2.size(2), xc2.size(1), xc2.size(0), -1).permute(1, 0, 2, 3).contiguous()
        #print(lstm_C_out2.shape)

        lstm_C_out2 = lstm_C_out2.view(-1, lstm_C_out2.size(3))
        tag_space = self.hidden2tag_1(lstm_C_out2)
        tag_space = self.hidden2tag_3(tag_space).view(-1, xc2.size(0))
        tag_space = tag_space.view(xc2.size(1), xc2.size(2), -1).permute(2, 0, 1).squeeze(0).contiguous()
        return tag_space,xp2

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)#Bottleneck
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)#Bottleneck
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)#Bottleneck
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == '__main__':
    resnet = resnet50()
    x = Variable(torch.randn(1, 3, 224, 224))
    print(resnet(x).size())
