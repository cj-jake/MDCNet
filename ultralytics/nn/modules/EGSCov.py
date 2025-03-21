import torch
from torch import nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EGSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 4
        # First two branches
        self.cv1a = Conv(c1, c_, k, s, None, g, act)
        self.cv1b = Conv(c1, c_, k, s, None, g, act)
        # Second two branches
        self.cv2a = Conv(c_, c_, 5, 1, None, c_, act)
        self.cv2b = Conv(c_, c_, 5, 1, None, c_, act)

    def channel_shuffle(self, x, groups):
        # Reshape and shuffle channels
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(batchsize, -1, height, width)

    def forward(self, x):
        # First pair
        x1a = self.cv1a(x)
        x1b = self.cv1b(x)
        x_concat1 = torch.cat((x1a, x1b), 1)
        x_shuffle1 = self.channel_shuffle(x_concat1, 2)

        # Second pair
        x2a = self.cv2a(x1a)
        x2b = self.cv2b(x1b)
        x_concat2 = torch.cat((x2a, x2b), 1)
        x_shuffle2 = self.channel_shuffle(x_concat2, 2)

        # Final concatenation and shuffle
        x_final = torch.cat((x_shuffle1, x_shuffle2), 1)
        output = self.channel_shuffle(x_final, 4)

        return output

if __name__ == '__main__':
    from torchviz import make_dot
    import torch

    # Instantiate your EGSConv model
    model = EGSConv(c1=3, c2=12, k=1, s=1, g=1, act=True)
    print(model)
    # Create a sample input tensor
    x = torch.randn(1, 3, 32, 32)  # Example input: batch size = 1, 16 channels, 32x32 spatial size

    # Forward pass
    y = model(x)

    # Visualize the computation graph
    graph = make_dot(y, params=dict(model.named_parameters()))
    graph.render("egsconv_visualization", format="png", cleanup=True)

    print("Graph saved as 'egsconv_visualization.png'.")
