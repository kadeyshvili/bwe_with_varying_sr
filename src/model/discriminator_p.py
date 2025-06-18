import torch.nn as nn
import torch.nn.functional as F


class SubMultiPeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size, stride, channels):
        super().__init__()
        self.period = period
        conv_layers = []
        norm = nn.utils.parametrizations.weight_norm
        for i in range(len(channels) - 1):
            conv_layers.append(nn.Sequential(norm(nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1],\
                                                                       kernel_size=(kernel_size, 1), stride=(stride, 1), \
                                                                        padding=(2, 0))), nn.LeakyReLU(0.1)))
        self.conv_blocks = nn.ModuleList(conv_layers)
        self.conv1 =  nn.Sequential(norm(nn.Conv2d(in_channels=channels[-1], out_channels=1024, kernel_size=(5, 1), padding="same")), \
                                    nn.LeakyReLU(0.1))
        self.res_conv = norm(nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=(3, 1),padding="same"))


    def make1d_to2d(self, x):
        if x.shape[-1] % self.period != 0:
            n_pad = self.period - (x.shape[-1] % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
        return x.view(x.shape[0], 1, x.shape[-1] // self.period, self.period)

    def forward(self, x):
        feats = []
        x = self.make1d_to2d(x)
        for layer in self.conv_blocks:
            x = layer(x)
            feats.append(x)
        x = self.conv1(x)
        feats.append(x)
        x = self.res_conv(x)
        feats.append(x)
        return x.flatten(-2, -1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods, kernel_size, stride, channels):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([SubMultiPeriodDiscriminator(period=period,kernel_size=kernel_size, \
                                                                             stride=stride,channels=channels)for period in periods])


    def forward(self, x_gt, x_gen):
        disc_outputs_gt = []
        disc_outputs_fake = []
        disc_features_gt = []
        disc_features_fake = []
        for disc in self.sub_discriminators:
            output_gt, features_list_gt = disc(x_gt)
            output_fake, features_list_fake = disc(x_gen)
            disc_outputs_gt.append(output_gt)
            disc_outputs_fake.append(output_fake)
            disc_features_gt.append(features_list_gt)
            disc_features_fake.append(features_list_fake)
        return disc_outputs_gt, disc_features_gt, disc_outputs_fake, disc_features_fake
