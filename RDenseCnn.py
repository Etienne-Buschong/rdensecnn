import torch
import torch.nn.functional as F
import torch.utils.data


GROWTH_RATE_MULTIPLIER = 4


class DenseLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.bn_1 = torch.nn.BatchNorm2d(in_channels)
        self.conv_1_1 = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_2 = torch.nn.BatchNorm2d(out_channels)
        self.conv_3_3 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn_1(x))
        x = self.conv_1_1(x)
        x = F.relu(self.bn_2(x))
        return self.conv_3_3(x)


class DenseBlock(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k):
        super(DenseBlock, self).__init__()
        self.dense_layers = torch.nn.ModuleList()
        channels = in_channels
        for i in range(num_layers_m):
            self.dense_layers.append(DenseLayer(channels, growth_rate_k))
            channels += growth_rate_k

    def forward(self, x):
        cat_input = x
        for dense_layer in self.dense_layers:
            layer_output = dense_layer(cat_input)
            cat_input = torch.cat([cat_input, layer_output], dim=1)
        return cat_input


class TransitionBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pool_kernel_size=2):
        super(TransitionBlock, self).__init__()
        self.bn_1 = torch.nn.BatchNorm2d(in_channels)
        self.conv_1_1 = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.avg_pooling = torch.nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = F.relu(self.bn_1(x))
        x = self.conv_1_1(x)
        return self.avg_pooling(x)


class ResidualDenseBlock(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k):
        super(ResidualDenseBlock, self).__init__()
        self.down_sample_fn = torch.nn.AvgPool2d(2)
        self.dense_block = DenseBlock(in_channels, num_layers_m, growth_rate_k)
        dense_channels_out = in_channels + num_layers_m * growth_rate_k
        self.transition_block = TransitionBlock(dense_channels_out, growth_rate_k * GROWTH_RATE_MULTIPLIER)

    def forward(self, x):
        residual = self.down_sample_fn(x)
        x = self.dense_block(x)
        x = self.transition_block(x)
        return x + residual


class RDenseCNN(torch.nn.Module):

    def __init__(self, num_channels, num_rd_blocks, num_layers_m, growth_rate_k, num_classes):
        super(RDenseCNN, self).__init__()
        base_res_block_channels = growth_rate_k * GROWTH_RATE_MULTIPLIER
        self.conv_3_3 = torch.nn.Conv2d(num_channels, base_res_block_channels, 3, padding=1)
        self.avg_pool = torch.nn.AvgPool2d(2)
        self.rd_blocks = torch.nn.ModuleList()
        for _ in range(num_rd_blocks):
            self.rd_blocks.append(ResidualDenseBlock(base_res_block_channels, num_layers_m, growth_rate_k))
        self.final_dense_block = DenseBlock(base_res_block_channels, num_layers_m, growth_rate_k)
        self.global_avg_pool = torch.nn.AvgPool2d(4)
        self.final_dense_block_out_channels = base_res_block_channels + num_layers_m * growth_rate_k
        self.fc_layer = torch.nn.Linear(self.final_dense_block_out_channels, num_classes)

    def forward(self, x):
        x = self.conv_3_3(x)
        x = self.avg_pool(x)
        for rd_block in self.rd_blocks:
            x = rd_block(x)
        x = self.final_dense_block(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, self.final_dense_block_out_channels * 1 * 1)
        return self.fc_layer(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())