
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ShuffleAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(ShuffleAttention, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Adjust the channel attention to output the same number of channels as the input
        self.channel_attn = nn.Conv2d(channel, channel, 1)  # Change to output `channel` instead of `channel/reduction`

        # 3x3 conv for spatial attention
        self.spatial_attn = nn.Conv2d(channel, channel, 3, padding=1)

        # Final attention layers for both space and channel (1x1 conv)
        self.final_attn = nn.Conv2d(channel * 2, channel, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention (based on global average pool)
        channel_weight = self.avg_pool(x)  # b, c, 1, 1
        channel_weight = self.channel_attn(channel_weight)  # b, c, 1, 1
        channel_weight = torch.sigmoid(channel_weight)  # apply sigmoid to normalize

        # Apply the channel attention
        x_channel = x * channel_weight.expand_as(x)  # Expand to match input size

        # Spatial attention (based on local context, using a 3x3 convolution)
        spatial_weight = self.spatial_attn(x)  # b, c, h, w
        spatial_weight = torch.sigmoid(spatial_weight)  # apply sigmoid to normalize

        # Apply the spatial attention
        x_spatial = x * spatial_weight

        # Concatenate both channel and spatial attention features
        x_attention = torch.cat([x_channel, x_spatial], dim=1)  # concatenate along channel axis

        # Final convolution to merge both attentions
        out = self.final_attn(x_attention)

        return out


if __name__ == '__main__':

        # Set channel to 256 (or any value divisible by 8) to avoid reshaping issues
        model = ShuffleAttention(channel=256, reduction=16)  # channel=256 works fine with G=8

        # 创建 TensorBoard Writer
        log_dir = 'runs/my_model_experiment'
        writer = SummaryWriter(log_dir)

        # 打印日志路径
        print(f"TensorBoard logs are being saved in: {log_dir}")

        # 创建一个虚拟输入（假设输入为 28x28 的单通道图像）
        dummy_input = torch.randn(1, 256, 28, 28)  # 输入尺寸 (batch_size, channels, height, width)

        # 将模型结构记录到 TensorBoard
        writer.add_graph(model, dummy_input)

        # 关闭 Writer
        writer.close()
