import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter


class innovative_layer(nn.Module):
    """An advanced version of the sa_layer module with Multi-head Attention and Dynamic Convolution"""

    def __init__(self, channel, num_heads=4):
        super(innovative_layer, self).__init__()
        self.num_heads = num_heads

        # Adaptive Pooling Layer for channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Parameters for Channel Attention
        self.cweight = Parameter(torch.zeros(1, channel // 4, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // 4, 1, 1))

        # Parameters for Spatial Attention
        self.sweight = Parameter(torch.zeros(1, channel //4, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // 4, 1, 1))

        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=channel, num_heads=num_heads)

        # Sigmoid and Group Normalization
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // 4, channel // 4)

        # Dynamic Convolution Layer (as an example of Dynamic Convolution)
        self.dynamic_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)

    def dynamic_conv_layer(self, x):
        """
        Dynamic convolution layer: the convolution kernel changes dynamically
        based on the input feature.
        """
        b, c, h, w = x.shape
        # Generate a dynamic convolution kernel based on the input size
        kernel = self.dynamic_conv.weight.view(c, c, 3, 3)
        kernel = kernel / kernel.sum(dim=(2, 3), keepdim=True)  # Normalize kernel
        out = F.conv2d(x, kernel, padding=1)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape input tensor
        x = x.reshape(b * 2, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # Channel Attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # Spatial Attention with Group Normalization
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # Concatenate and Apply Multi-head Attention
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        # Apply Multihead Attention (self-attention)
        out = out.view(b, c, h * w).permute(2, 0, 1)  # Prepare for Multihead Attention
        out, _ = self.attn(out, out, out)  # Self-attention
        out = out.permute(1, 2, 0).view(b, c, h, w)

        # Apply Dynamic Convolution
        out = self.dynamic_conv_layer(out)

        return out



if __name__ == '__main__':
    # # 示例输入
    # x = torch.randn(8, 512, 32, 32)  # Batch size 8, 512 channels, 32x32 feature map
    #
    # # 创建并应用多尺度ShuffleAttention
    # model = ShuffleAttention(channel=512, G=8)
    # output = model(x)

    # 输出尺寸
    # print(output.shape)  # [8, 512*3, 32, 32]，通道数变为原来的3倍（因为拼接了3个尺度的特征）
    # 初始化模型和输入
    model =innovative_layer(channel=512, groups=8)
    x = torch.randn(8, 512, 32, 32)  # 示例输入

    y = model(x)
    # 这三种方式都可以
    # g = make_dot(y)
    # g=make_dot(y, params=dict(model.named_parameters()))
    # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    # 这两种方法都可以
    # g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
    # g.render('espnet_model', view=False)  # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
    output = model(x)

    # 输出尺寸
    print(output.shape)  # [8, 512*3, 32, 32]，通道数变为原来的3倍（因为拼接了3个尺度的特征）
    # 初始化 TensorBoard SummaryWriter
    writer = SummaryWriter('runs/MultiscaleShuffleAttention')

    # 记录计算图
    writer.add_graph(model, x)

    # 记录一些标量（比如损失值）
    # writer.add_scalar('Loss/train', loss_value, epoch)

    # 关闭 writer
    writer.close()

    # 提示用户启动 TensorBoard
    print("模型计算图已记录，运行以下命令来查看：")
    print("tensorboard --logdir=runs")
