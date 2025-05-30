import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.3):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.Dropout2d(p=dropout))
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False), nn.Dropout2d(p=dropout))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def shared_dropout2d(self, x):
        if not self.training or self.dropout == 0:
            return x
        # Create shared dropout mask with shape [1, C, 1, 1] and broadcast to batch
        mask = x.new_ones((1, x.size(1), 1, 1))
        mask = F.dropout2d(mask, p=self.dropout, training=True)
        return x * mask  # broadcast across batch

    def forward(self, x):
        out = self.conv1(x)
        out = self.shared_dropout2d(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.shared_dropout2d(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.drop = nn.Dropout2d(p=self.dropout)
        # self.drop2 = nn.Dropout(p=self.dropout)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout=self.dropout)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.drop(out)
        out = self.layer1(out)
        # out = self.drop(out)
        out = self.layer2(out)
        # out = self.drop(out)
        out = self.layer3(out)
        # out = self.drop(out)
        out = self.layer4(out)
        # out = self.drop(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetReduce(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout=0.5):
        super(ResNetReduce, self).__init__()
        self.in_planes = 64
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.drop = nn.Dropout2d(p=self.dropout)
        # self.drop2 = nn.Dropout(p=self.dropout)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2, dropout=self.dropout)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.drop(out)
        out = self.layer1(out)
        # out = self.drop(out)
        out = self.layer2(out)
        # out = self.drop(out)
        out = self.layer3(out)
        # out = self.drop(out)
        out = self.layer4(out)
        # out = self.drop(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18Reduce(dropout=0.5, num_classes=10):
    return ResNetReduce(BasicBlock, [2, 2, 2, 2], dropout=dropout, num_classes=num_classes)

# def ResNet18(dropout=0.5, num_classes=10):
#     return ResNet(BasicBlock, [2, 2, 2, 2], dropout=dropout, num_classes=num_classes)

def ResNet18(dropout=0.5, num_classes=10):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class MLP(nn.Module):
    def __init__(self, n_hidden_nodes, n_hidden_layers, activation, keep_rate=0):
        super(Net, self).__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,
                                   n_hidden_nodes)
        self.fc1_drop = torch.nn.Dropout(1 - keep_rate)
        if n_hidden_layers == 2:
            self.fc2 = torch.nn.Linear(n_hidden_nodes,
                                       n_hidden_nodes)
            self.fc2_drop = torch.nn.Dropout(1 - keep_rate)

        self.out = torch.nn.Linear(n_hidden_nodes, N_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS)
        if self.activation == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc1_drop(x)
        if self.n_hidden_layers == 2:
            if self.activation == "sigmoid":
                x = sigmoid(self.fc2(x))
            elif self.activation == "relu":
                x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc2_drop(x)
        return torch.nn.functional.log_softmax(self.out(x))