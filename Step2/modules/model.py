import torch
import torch.nn as nn

#残差ブロックを作成するクラス
class Resblock(nn.Module):
  def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
    super(Resblock, self).__init__()

    self.expansion = 4

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

    self.identity_downsample = identity_downsample
    self.relu = nn.ReLU()

  def forward(self, x):
    identity = x.clone()

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)
    x = x + identity

    x = self.relu(x)

    return x


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes):
    super(ResNet, self).__init__()

    self.in_channels = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    self.conv2_x = self._make_layer(block, layers[0], first_conv_out_channels=64, stride=1)
    self.conv3_x = self._make_layer(block, layers[1], first_conv_out_channels=128, stride=2)
    self.conv4_x = self._make_layer(block, layers[2], first_conv_out_channels=256,stride=2)
    self.conv5_x = self._make_layer(block, layers[3], first_conv_out_channels=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*4, num_classes)


  def forward(self, x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv2_x(x)
    x = self.conv3_x(x)
    x = self.conv4_x(x)
    x = self.conv5_x(x)
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x

    # 例 conv2_x のチャネル数の変化
    # in : 64
    # block(in_channels=64, first_conv_out_channels=64) out : 256
    # block(in_channels=256, first_conv_out_channels=64) out : 256
    # block(in_channels=256, first_conv_out_channels=64) out : 256
    # out : 256

  def _make_layer(self, block, num_resblocks, first_conv_out_channels, stride):
    identity_downsample = None
    layers = []

    # size 変更したいあるいは
    # チャネル数の変更が起こる場合残差接続でサイズとチャネル数が一致しなくなってしまうため
    # identity に conv層 を追加する。
    if stride != 1 or self.in_channels != first_conv_out_channels*4:
      identity_downsample = nn.Conv2d(self.in_channels, first_conv_out_channels*4, kernel_size=1, stride=stride)
    layers.append(block(self.in_channels, first_conv_out_channels, identity_downsample, stride))

    # 二つ目以降の残差ブロックでは常に、残差ブロックの入力チャネル数 in_channels が
    # その最初のconv層の出力チャネル数 first_conv_out_channels の4倍になっていることに注意
    self.in_channels = first_conv_out_channels*4

    for i in range(num_resblocks - 1):
      layers.append(block(self.in_channels, first_conv_out_channels))

    return nn.Sequential(*layers)

def get_ResNet50(block, num_classes):
  return ResNet(block, [3,4,6,3], num_classes)

def get_ResNet101(block, num_classes):
  return ResNet(block, [3,4,23,3], num_classes)

def get_ResNet152(block, num_classes):
  return ResNet(block, [3,8,36,3], num_classes)

