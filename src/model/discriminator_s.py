import torch.nn as nn


class SubMultiScaleDiscriminator(nn.Module):
    def __init__(self,  kernel_sizes, strides, groups, channels, use_spectral=False):
        super().__init__()
        layers = []
        norm = nn.utils.spectral_norm if use_spectral else nn.utils.parametrizations.weight_norm
        for i in range(len(kernel_sizes)):
            layers.append(nn.Sequential(norm(nn.Conv1d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_sizes[i], \
                                                              stride=strides[i], groups=groups[i], padding=(kernel_sizes[i] - 1) // 2)), nn.LeakyReLU(0.1)))

        layers.append(norm(nn.Conv1d( in_channels=channels[-1], out_channels=1, kernel_size=3, stride=1, padding=1)))

        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return x, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_blocks, kernel_sizes, strides, groups, channels):
        super().__init__()
        self.discriminators = nn.ModuleList([SubMultiScaleDiscriminator( kernel_sizes=kernel_sizes, strides=strides, groups=groups, \
                                                                        channels=channels, use_spectral=(i == 0)) for i in range(num_blocks)])
        self.pooling = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2),nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, x_gt, x_fake):
        disc_gt_outputs = []
        disc_gt_features = []
        disc_fake_outputs = []
        disc_fake_features = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x_gt = self.pooling[i-1](x_gt)
                x_fake = self.pooling[i-1](x_fake)
            output_gt, features_list_gt = disc(x_gt)
            disc_gt_outputs.append(output_gt)
            disc_gt_features.append(features_list_gt)
            output_fake, features_list_fake = disc(x_fake)
            disc_fake_outputs.append(output_fake)
            disc_fake_features.append(features_list_fake)
        return disc_gt_outputs, disc_gt_features, disc_fake_outputs, disc_fake_features
    