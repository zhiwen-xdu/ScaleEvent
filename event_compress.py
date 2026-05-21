import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv2d(nn.Module):
    """
    自定义的稀疏卷积层
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 卷积核
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        :param x: 输入稀疏张量，形状为 (N, C, H, W)，N为batch大小，C为通道数，H和W为图像的宽高
        :return: 卷积后的输出
        """
        # 通过稀疏矩阵乘法实现稀疏卷积
        return F.conv2d(x.to_dense(), self.weight, self.bias, stride=self.stride, padding=self.padding)


class SparseEventCompressionNet(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=512, out_channels=128):
        super(SparseEventCompressionNet, self).__init__()

        # 编码器部分：稀疏卷积层
        self.conv1 = SparseConv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = SparseConv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = SparseConv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1)

        # 解码器部分：逆稀疏卷积层（反卷积）
        self.deconv1 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 编码器部分：稀疏卷积操作
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        # 解码器部分：逆卷积操作
        x = self.deconv1(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = F.relu(x)

        x = self.deconv3(x)

        return x


# 示例代码：创建一个输入形状为 HxWx256 的稀疏事件张量
H, W = 64, 64
in_channels = 256

# 创建稀疏张量输入（这里只是一个简单的例子，真实应用中需要使用稀疏格式的数据）
x = torch.randn(H, W, in_channels).to_sparse()

print(x)

# 实例化网络
model = SparseEventCompressionNet()
# 将稀疏事件张量传入网络进行压缩
output = model(x)

print("输出形状:", output.shape)
