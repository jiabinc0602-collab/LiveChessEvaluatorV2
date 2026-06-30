import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        # not for this resnet we are not downsampling, so stride is always 1

        # conv layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # conv layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        residual = x
        # pass through layer 1
        out = self.relu(self.bn1(self.conv1(x)))
        # pass through layer 2
        out = self.bn2(self.conv2(out))
        # residual connection
        out += self.shortcut(residual)
        # apply ReLU activation
        out = self.relu(out)

        return out
    



class ChessResNet(nn.Module):
    def __init__(self, input_channels=20, num_blocks=5, num_classes=1):
        super().__init__()
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_blocks = self._make_layer(64, 64, num_blocks)

        # Compressing values with 1x1 convolution to reduce the number of channels before flattening
        self.value_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.relu_value = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layer for output
        self.fc = nn.Sequential(
            nn.Linear(2 * 8 * 8, 64),  # Flattened size after conv layers
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.Sigmoid()  # Use Sigmoid for binary output (0 to 1)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ResBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolutional layer
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Pass through residual blocks
        out = self.res_blocks(out)

        # Compressing values with 1x1 convolution
        out = self.relu_value(self.value_bn(self.value_conv(out)))
        
        # Flatten the output for the fully connected layer
        out = self.flatten(out)
        
        # Fully connected layers
        out = self.fc(out)
        
        return out