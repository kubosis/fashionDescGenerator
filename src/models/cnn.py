import torch
import torch.nn as nn

class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvSilu(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvSilu(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class BottleneckCSP(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvSilu(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvSilu(in_channels, hidden_channels, 1, 1)
        self.conv3 = ConvSilu(2 * hidden_channels, out_channels, 1, 1)

        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0)
            for _ in range(n)
        ])

        self.bn = nn.BatchNorm2d(2 * hidden_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        y1 = self.m(self.conv1(x))  # processed path
        y2 = self.conv2(x)  # shortcut path
        y = torch.cat((y1, y2), dim=1)
        return self.conv3(self.act(self.bn(y)))

class CNN(nn.Module):
    """
    Basic configurable CNN
    """
    def __init__(self, w: int, h: int, c: int, out_dim: int):
        """
        :param h: (int) input image height
        :param w: (int) input image width
        :param c: (int) input image channels

        :param hidden_dim: (int) dimension of hidden layers
        :param hidden_layers: (int) number of hidden layers
        :param out_dim: (int) output vector dimension
        """

        # TODO: Create a convolution neural network that:
        # extracts the features from the image by sliding the convolutional window over the input image
        # (use torch.nn.Conv2d. use more layers of this type, always halving the h, w and doubling the number of channels)
        # use residual connections between the convolutional

        super().__init__()
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (h x w x c)
        :return:
        """
        pass
