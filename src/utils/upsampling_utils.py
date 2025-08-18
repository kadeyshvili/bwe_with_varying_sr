from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import  spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Literal



def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels,
        conditional_channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)
        self.conditional_conv = nn.ModuleList([
            weight_norm(Conv1d(conditional_channels, channels, 1, 1, dilation=1,
                               padding=get_padding(1, 1))),
            weight_norm(Conv1d(conditional_channels, channels, 1, 1, dilation=1,
                               padding=get_padding(1, 1))),
            weight_norm(Conv1d(conditional_channels, channels, 1, 1, dilation=1,
                               padding=get_padding(1, 1)))
        ])
        self.conditional_conv.apply(init_weights)


    def forward(self, x, condition):
        for c1, c2, con in zip(self.convs1, self.convs2, self.conditional_conv):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            condition_proj = con(condition)
            xt = xt + condition_proj
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        super(ResBlock2, self).__init__()
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class AddSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.add.add(x, self.net(x))


class ConcatSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


def build_block(
        inner_width,
        block_depth,
        mode: Literal["unet_k3_2d", "waveunet_k5"],
        norm
):
    if mode == "unet_k3_2d":
        return nn.Sequential(
            *[
                AddSkipConn(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        norm(
                            nn.Conv2d(
                                inner_width,
                                inner_width,
                                3,
                                padding=1,
                                bias=True,
                            )
                        ),
                    )
                )
                for _ in range(block_depth)
            ]
        )
    elif mode == "waveunet_k5":
        return nn.Sequential(
            *[
                AddSkipConn(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        norm(
                            nn.Conv1d(
                                inner_width,
                                inner_width,
                                5,
                                padding=2,
                                bias=True,
                            )
                        ),
                    )
                )
                for _ in range(block_depth)
            ]
        )
    else:
        raise NotImplementedError


class MultiScaleResnet(nn.Module):
    def __init__(
        self,
        block_widths,
        block_depth,
        in_width=1,
        out_width=1,
        downsampling_conv=True,
        upsampling_conv=True,
        concat_skipconn=True,
        scale_factor=4,
        mode: Literal["waveunet_k5"] = "waveunet_k5",
        norm_type: Literal["weight", "spectral", "id"] = "id"
    ):
        super().__init__()
        norm = dict(
            weight=weight_norm, spectral=spectral_norm, id=lambda x: x
        )[norm_type]
        self.in_width = in_width
        self.out_dims = out_width
        net = build_block(block_widths[-1], block_depth, mode, norm)
        for i in range(len(block_widths) - 1):
            width = block_widths[-2 - i]
            inner_width = block_widths[-1 - i]
            if downsampling_conv:
                downsampling = norm(nn.Conv1d(
                    width, inner_width, scale_factor, scale_factor, 0
                ))
            else:
                downsampling = nn.Sequential(
                    nn.AvgPool1d(scale_factor, scale_factor),
                    norm(nn.Conv1d(width, inner_width, 1)),
                )
            if upsampling_conv:
                upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    norm(nn.Conv1d(inner_width, width, 1)),
                )
            else:
                upsampling = norm(nn.ConvTranspose1d(
                    inner_width, width, scale_factor, scale_factor, 0
                ))
            net = nn.Sequential(
                build_block(width, block_depth, mode, norm),
                downsampling,
                net,
                upsampling,
                build_block(width, block_depth, mode, norm),
            )
            if concat_skipconn:
                net = nn.Sequential(
                    ConcatSkipConn(net),
                    norm(nn.Conv1d(width * 2, width, 1)),
                )
            else:
                net = AddSkipConn(net)
        self.net = nn.Sequential(
            norm(nn.Conv1d(in_width, block_widths[0], 5, padding=2)),
            net,
            norm(nn.Conv1d(block_widths[0], out_width, 5, padding=2)),
        )

    def forward(self, x):
        assert x.shape[1] == self.in_width, (
            "%d-dimensional condition is assumed" % self.in_width
        )
        return self.net(x)


class MultiScaleResnet2d(nn.Module):
    def __init__(
        self,
        block_widths,
        block_depth,
        in_width=1,
        out_width=1,
        downsampling_conv=True,
        upsampling_conv=True,
        concat_skipconn=True,
        scale_factor=4,
        mode: Literal["unet_k3_2d"] = "unet_k3_2d",
        norm_type: Literal["weight", "spectral", "id"] = "id",
    ):
        super().__init__()
        self.in_width = in_width
        self.out_dims = out_width
        norm = dict(weight=weight_norm, spectral=spectral_norm, id=lambda x: x)[norm_type]
        net = build_block(block_widths[-1], block_depth, mode, norm)
        for i in range(len(block_widths) - 1):
            width = block_widths[-2 - i]
            inner_width = block_widths[-1 - i]
            if downsampling_conv:
                downsampling = norm(
                    nn.Conv2d(
                        width, inner_width, scale_factor, scale_factor, 0
                    )
                )
            else:
                downsampling = nn.Sequential(
                    nn.AvgPool2d(scale_factor, scale_factor),
                    norm(
                        nn.Conv2d(width, inner_width, 1),
                    ),
                )
            if upsampling_conv:
                upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    norm(nn.Conv2d(inner_width, width, 1)),
                )
            else:
                upsampling = norm(
                    nn.ConvTranspose2d(
                        inner_width, width, scale_factor, scale_factor, 0
                    )
                )
            net = nn.Sequential(
                build_block(width, block_depth, mode, norm),
                downsampling,
                net,
                upsampling,
                build_block(width, block_depth, mode, norm),
            )
            if concat_skipconn:
                net = nn.Sequential(
                    ConcatSkipConn(net),
                    norm(nn.Conv2d(width * 2, width, 1)),
                )
            else:
                net = AddSkipConn(net)
        self.net = nn.Sequential(
            norm(nn.Conv2d(in_width, block_widths[0], 3, padding=1)),
            net,
            norm(nn.Conv2d(block_widths[0], out_width, 3, padding=1)),
        )

    def forward(self, x):
        assert x.shape[1] == self.in_width, (
            "%d-dimensional condition is assumed" % self.in_width
        )
        # padding to across spectral dimension to be divisible by 16
        # (max depth assumed to be 4)
        pad = 16 - x.shape[-2] % 16
        shape = x.shape
        padding = torch.zeros((shape[0], shape[1], pad, shape[3])).to(x)
        x1 = torch.cat((x, padding), dim=-2)
        return self.net(x1)[:, :, : x.shape[2]]



class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations, norm_type: Literal["weight", "spectral", "id"] = "weight",):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm, id=lambda x: x)[norm_type]
        self.blocks = nn.ModuleList(nn.ModuleList([None for _ in range(len(dilations[0]))]) for _ in range(len(dilations)))
        for i in range(len(dilations)):
            for j in range(len(dilations[0])):
                self.blocks[i][j] = nn.Sequential(nn.LeakyReLU(0.1), self.norm(nn.Conv1d(in_channels=channels, out_channels=channels, \
                                                                                    kernel_size=kernel_size, dilation=dilations[i][j],padding="same")))
    
    def forward(self, x):
        for i in range(len(self.blocks)):
            res = x
            for j in range(len(self.blocks[i])):
                x = self.blocks[i][j](x)
            x = x + res
        return x
    
class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(channels=channels, kernel_size=kernel_sizes[i], dilations=dilations[i]) \
                                         for i in range(len(kernel_sizes))])

    def forward(self, x):
        result = self.res_blocks[0](x)
        for block in self.res_blocks[1:]:
            result += block(x)
        return result
    

class UpsampleTwice(torch.nn.Module):
    def __init__(
            self, 
            initial_channels,
            upsample_rates, 
            upsample_kernel_sizes,
            norm_type: Literal["weight", "spectral", "id"] = "id",
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm, id=lambda x: x)[norm_type]

        self.upsample_blocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsample_blocks.append(
                self.norm(
                    nn.ConvTranspose1d(
                        initial_channels,
                        initial_channels,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
    def forward(self, x):
        out = x
        for i in range(len(self.upsample_blocks)):
            out = self.upsample_blocks[i](out)
        return out
    

class UpsampleTwiceWithMRF(torch.nn.Module):
    def __init__(
            self, 
            initial_channels,
            upsample_rates, 
            upsample_kernel_sizes,
            kernel_sizes_mrf,
            dilations_mrf,
            norm_type: Literal["weight", "spectral", "id"] = "weight",
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm, id=lambda x: x)[norm_type]

        self.upsample_blocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsample_blocks.append(
                self.norm(
                    nn.ConvTranspose1d(
                        initial_channels,
                        initial_channels,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        kernel_sizes = kernel_sizes_mrf 
        dilations = dilations_mrf 
        self.mrfs = nn.ModuleList()
        for _ in range(len(upsample_rates)):
            self.mrfs.append(MRF(initial_channels, kernel_sizes, dilations))
    
    def forward(self, x):
        out = x
        for i in range(len(self.upsample_blocks)):
            out = self.upsample_blocks[i](out)
            out = self.mrfs[i](out) 
        return out


Linear = nn.Linear
silu = F.silu
relu = F.relu

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class BSFT(nn.Module):
    def __init__(self, nhidden, out_channels):
        super().__init__()
        self.mlp_shared = nn.Conv1d(2, nhidden, kernel_size=3, padding=1)

        self.mlp_gamma = Conv1d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = Conv1d(nhidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x, band):
        actv = silu(self.mlp_shared(band))

        gamma = self.mlp_gamma(actv).unsqueeze(-1)
        beta = self.mlp_beta(actv).unsqueeze(-1)
        out = x * (1 + gamma) + beta

        return out
    


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels):
        super(FourierUnit, self).__init__()

        self.conv_layer = Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                 kernel_size=1, padding=0, bias=False)
        self.bsft = BSFT(bsft_channels, out_channels * 2)
        self.n_fft=1024
        self.hop_size=256
        self.win_size=1024
        self.hann_window=torch.hann_window(self.win_size)

    def forward(self, x, band):
        batch = x.shape[0]

        x = x.view(-1, x.size()[-1])

        ffted = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window.to(x.device),
                          center=True, normalized=True, onesided=True, return_complex=False)
        ffted = ffted.permute(0, 3, 1, 2).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[2:])

        ffted = relu(self.bsft(ffted, band))
        ffted = self.conv_layer(ffted)

        ffted = ffted.view((-1, 2,) + ffted.size()[2:]).permute(0, 2, 3, 1).contiguous()
        real, imag = ffted[..., 0], ffted[..., 1]
        ffted_complex = torch.complex(real, imag)

        output = torch.istft(ffted_complex, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window.to(ffted_complex.device),
                            center=True, normalized=True, onesided=True)

        output = output.view(batch, -1, x.size()[-1])
        return output

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels):
        super(SpectralTransform, self).__init__()
        self.conv1 = Conv1d(
            in_channels, out_channels // 2, kernel_size=1, bias=False)

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, bsft_channels)

        self.conv2 = Conv1d(
            out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x, band):
        x = silu(self.conv1(x))
        output = self.fu(x, band)
        output = self.conv2(x + output)

        return output
    

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, padding=1):
        super(FFC, self).__init__()

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        self.convl2l = Conv1d(in_cl, out_cl, kernel_size, padding=padding, bias=False)
        self.convl2g = Conv1d(in_cl, out_cg, kernel_size, padding=padding, bias=False)
        self.convg2l = Conv1d(in_cg, out_cl, kernel_size, padding=padding, bias=False)
        self.convg2g = SpectralTransform(in_cg, out_cg, bsft_channels)

    def forward(self, x_l, x_g, band):
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g, band)

        return out_xl, out_xg


class NUWaveBlock(nn.Module):
    def __init__(self, residual_channels, bsft_channels,):
        super().__init__()
        self.input_projection = Conv1d(2, residual_channels, 1)
        self.ffc1 = FFC(residual_channels, 2*residual_channels, bsft_channels, kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1) # STFC

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        self.output_projectio2 = Conv1d(residual_channels, 1, 1)


    def forward(self, initial_x, reference_x, band):
        initial_x = initial_x.squeeze(1)
        reference_x = reference_x.squeeze(1)
        x = torch.stack((initial_x, reference_x), dim=1)

        x = self.input_projection(x)

        y_l, y_g = torch.split(x, [x.shape[1] - self.ffc1.global_in_num, self.ffc1.global_in_num], dim=1)
        y_l, y_g = self.ffc1(y_l, y_g, band)
        gate_l, filter_l = torch.chunk(y_l, 2, dim=1)
        gate_g, filter_g = torch.chunk(y_g, 2, dim=1)
        gate, filter = torch.cat((gate_l, gate_g), dim=1), torch.cat((filter_l, filter_g), dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        out = self.output_projectio2(residual)
        return out



class HiFiUpsampling(torch.nn.Module):
    def __init__(
            self,
            resblock="2",
            upsample_initial_channel=128,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            conv_pre_kernel_size=1,
            input_channels=513,
            conditional_channels=513,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type
        self.num_kernels = len(resblock_kernel_sizes)
        self.upsample_rates = [8, 8, 2, 2]  
        self.upsample_kernel_sizes = [r * 2 for r in self.upsample_rates]
        self.num_upsamples = len(self.upsample_rates)
        
        self.make_conv_pre(
            input_channels,
            upsample_initial_channel,
            conv_pre_kernel_size
        )

        self.ups = None
        self.resblocks = None
        self.out_channels = self.make_resblocks(
            resblock,
            self.upsample_rates,
            self.upsample_kernel_sizes,
            upsample_initial_channel,
            conditional_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )

    def make_conv_pre(self, input_channels, upsample_initial_channel, kernel_size):
        assert kernel_size % 2 == 1
        self.conv_pre = self.norm(
            nn.Conv1d(
                input_channels, upsample_initial_channel, kernel_size, 1, padding=kernel_size // 2
            )
        )
        self.conv_pre.apply(init_weights)

    def make_resblocks(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        conditional_channels,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        resblock = (
            ResBlock1 if resblock == "1" else ResBlock2
        )

        self.ups = nn.ModuleList()
        self.cond_ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                self.norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            self.cond_ups.append(
                nn.ConvTranspose1d(
                    conditional_channels,
                    conditional_channels,
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        

        ch = None
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock(ch, conditional_channels, k, d, norm_type=self.norm_type)
                )
        self.ups.apply(init_weights)
        return ch

    def forward(self, x, condition):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            condition = F.leaky_relu(condition, LRELU_SLOPE)
            condition = self.cond_ups[i](condition)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, condition)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, condition)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        return x




class SpectralUNet(nn.Module):
    def __init__(
            self,
            block_widths=(8, 16, 24, 32, 64),
            block_depth=5,
            positional_encoding=True,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.norm_type = norm_type
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        in_width = 1
        out_width = block_widths[0] 

        self.net = MultiScaleResnet2d(
            block_widths,
            block_depth,
            scale_factor=2,
            in_width=in_width,
            out_width=out_width,
            norm_type=norm_type,
        )
        self.post_conv_2d = nn.Sequential(
            norm(nn.Conv2d(out_width, 1, 1, padding=0)),
        )

        self.post_conv_1d = nn.Sequential(
            norm(nn.Conv1d(513, 128, 1, 1, padding=0)),
        )

    def forward(self, magnitude):
        net_input = magnitude.unsqueeze(1)
        
        out = self.net(net_input)
        out = self.post_conv_2d(out)
        
        out = out.squeeze(1)
        
        out_magnitude = self.post_conv_1d(out)
        return out_magnitude
    
