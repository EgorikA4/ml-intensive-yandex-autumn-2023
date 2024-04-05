import torch
import torch.nn as nn
import torch.optim as optim

# Определение блока плотных связей
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1) for i in range(num_layers)])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))
        return torch.cat(features, 1)

# Определение слоя перехода
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))

# Определение архитектуры DenseNet
class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=12, num_classes=10):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1)
        self.dense1 = self._make_dense_block(2 * growth_rate, num_blocks[0], growth_rate)
        self.trans1 = self._make_transition_layer(2 * growth_rate, 0.5)
        self.dense2 = self._make_dense_block(2 * growth_rate, num_blocks[1], growth_rate)
        self.trans2 = self._make_transition_layer(2 * growth_rate, 0.5)
        self.dense3 = self._make_dense_block(2 * growth_rate, num_blocks[2], growth_rate)
        self.trans3 = self._make_transition_layer(2 * growth_rate, 0.5)
        self.dense4 = self._make_dense_block(2 * growth_rate, num_blocks[3], growth_rate)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * growth_rate, num_classes)

    def _make_dense_block(self, in_channels, num_layers, growth_rate):
        layers = []
        for _ in range(num_layers):
            layers.append(DenseBlock(in_channels, growth_rate, 4))
            in_channels += growth_rate * 4
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_channels, compression):
        out_channels = int(in_channels * compression)
        return TransitionLayer(in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Создание экземпляра DenseNet
num_blocks = [6, 12, 24, 16]
growth_rate = 32
model = DenseNet(num_blocks, growth_rate, 1)
