# util/model_utils.py
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    A single temporal block in the TCN architecture.
    Consists of two convolutional layers with weight normalization,
    dropout, and residual connections.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # ensure the residual connection matches dimensions
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        return out + res  # residual connection


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for multichannel time series classification.

    Input: (batch_size, in_channels, seq_length)
    Output: (batch_size, num_classes)
    """
    def __init__(self, in_channels=64, num_classes=8, num_levels=4,
                 kernel_size=3, dropout=0.2, channels=None):
        super(TemporalConvNet, self).__init__()
        # default channels per layer if not provided
        if channels is None:
            channels = [64, 64, 128, 128][:num_levels]

        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else channels[i-1]
            out_ch = channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x: batch x channels x seq_len
        y = self.network(x)
        # take last time step features for classification
        out = y[:, :, -1]
        return self.classifier(out)


def build_model(in_channels=64, num_classes=8, device=None):
    """
    Utility to build and return a TCN model on the given device.

    Args:
        in_channels: number of input channels (64 spatial dim)
        num_classes: number of output classes (7 or 8)
        device: torch device (e.g., 'cuda' or 'cpu')
    Returns:
        model: nn.Module moved to device
    """
    model = TemporalConvNet(in_channels=in_channels, num_classes=num_classes)
    if device:
        model = model.to(device)
    return model
