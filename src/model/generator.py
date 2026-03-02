from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import  spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import src.utils.upsampling_utils as upsampling_utils
import librosa
from src.model.melspec import MelSpectrogram
from src.utils.sr_utils import get_sr_ratio, get_intermediate_sr, create_band_mask, get_num_blocks



def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value=None,
        adanorm_num_embeddings=None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x, cond_embedding_id=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

mel_cache = {}
inv_mel_cache = {}
window_cache = {}

def stft(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin=0.0,
    fmax=None,
    center=True,
):
    global mel_cache, inv_mel_cache, window_cache
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    device = y.device

    if fmax is None:
        fmax = float(sampling_rate) / 2

    cache_key = f"{sampling_rate}_{n_fft}_{num_mels}_{fmin}_{fmax}_{win_size}_{device}"
    if cache_key not in mel_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis = torch.from_numpy(mel).float().to(device)
        inv_basis = mel_basis.pinverse()

        mel_cache[cache_key] = mel_basis
        inv_mel_cache[cache_key] = inv_basis
        window_cache[cache_key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_cache[cache_key].to(y.device)
    inv_basis = inv_mel_cache[cache_key].to(y.device)
    window = window_cache[cache_key]
    spec_complex = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        return_complex=True,
    )  # (B, F, T)
    return spec_complex

def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()



class HiFiPlusGenerator(torch.nn.Module):
    def __init__(
        self,
        hifi_resblock="1",
        hifi_upsample_rates=(8, 8, 2, 2),
        hifi_upsample_kernel_sizes=(16, 16, 4, 4),
        hifi_upsample_initial_channel=128,
        hifi_resblock_kernel_sizes=(3, 7, 11),
        hifi_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        hifi_input_channels=128,
        hifi_conv_pre_kernel_size=1,

        upsample_block_rates=[2, 2],
        upsample_init_channels = 1,
        upsample_block_kernel_sizes=[4, 4], 
        kernel_sizes_mrf = [3, 5, 7],
        dilations_mrf = [
            [[1, 3, 5], [1, 3, 5]],
            [[1, 3], [1, 3]],
            [[1], [1]]
        ],

        residual_channels=64,
        bsft_channels=64,
        nwstack1_blocks=1,
        nwstack2_blocks=1,

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,


        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type

        self.use_spectralunet = use_spectralunet
        self.use_waveunet = use_waveunet

        self.use_skip_connect = use_skip_connect
        self.upsampling_block1 = upsampling_utils.UpsampleTwice(upsample_init_channels, upsample_block_rates, upsample_block_kernel_sizes)
        self.upsampling_block2 = upsampling_utils.UpsampleTwice(upsample_init_channels, upsample_block_rates, upsample_block_kernel_sizes)
        self.upsampling_block_x3 = upsampling_utils.UpsampleThreeTimes(upsample_init_channels)
        self.nw_stack1 = upsampling_utils.NUWaveStack(residual_channels, bsft_channels, n_blocks=nwstack1_blocks)
        self.nw_stack2 = upsampling_utils.NUWaveStack(residual_channels, bsft_channels, n_blocks=nwstack2_blocks)
        if kernel_sizes_mrf is not None:
            self.upsampling_block = upsampling_utils.UpsampleTwiceWithMRF(hifi_upsample_initial_channel, upsample_block_rates, \
                                                               upsample_block_kernel_sizes, kernel_sizes_mrf, dilations_mrf)
        self.hifi = upsampling_utils.HiFiUpsampling(
            resblock=hifi_resblock,
            upsample_initial_channel=hifi_upsample_initial_channel,
            resblock_kernel_sizes=hifi_resblock_kernel_sizes,
            resblock_dilation_sizes=hifi_resblock_dilation_sizes,
            input_channels=hifi_input_channels,
            conv_pre_kernel_size=hifi_conv_pre_kernel_size,
            norm_type=norm_type,
        )
        ch = self.hifi.out_channels

        if self.use_spectralunet:
            self.spectralunet = upsampling_utils.SpectralUNet(
                block_widths=spectralunet_block_widths,
                block_depth=spectralunet_block_depth,
                positional_encoding=spectralunet_positional_encoding,
                norm_type=norm_type,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=513, 
                out_channels=128, 
                kernel_size=1,
                stride=1,
                padding=0
            )

        if self.use_waveunet:
            self.waveunet = upsampling_utils.MultiScaleResnet(
                waveunet_block_widths,
                waveunet_block_depth,
                mode="waveunet_k5",
                out_width=ch,
                in_width=ch,
                norm_type=norm_type
            )

        self.waveunet_skip_connect = None
        self.spectralmasknet_skip_connect = None
        if self.use_skip_connect:
            self.make_waveunet_skip_connect(ch)

        self.conv_post = None
        self.make_conv_post(ch)

    def make_waveunet_skip_connect(self, ch):
        self.waveunet_skip_connect = self.norm(nn.Conv1d(ch, ch, 1, 1))
        self.waveunet_skip_connect.weight.data = torch.eye(ch, ch).unsqueeze(-1)
        self.waveunet_skip_connect.bias.data.fill_(0.0)


    def make_conv_post(self, ch):
        self.conv_post = self.norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post.apply(upsampling_utils.init_weights)

    def apply_spectralunet(self, x_reference):
        if self.use_spectralunet:
            orig_length = x_reference.shape[-1]
            pad_size = (
                closest_power_of_two(orig_length) - orig_length
            )
            if pad_size > 0:
                x = torch.nn.functional.pad(x_reference, (0, pad_size))
            else:
                x = x_reference
        
            x_mag = self.spectralunet(x)
            x_mag = x_mag[..., :orig_length]
        else:
            x = x_reference.squeeze(1)
        return x_mag

    def apply_waveunet(self, x):
        x_a = x
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x



    def forward(self, x_reference):
        x = self.apply_spectralunet(x_reference)
        x = self.hifi(x)
        if self.use_waveunet:
            x = self.apply_waveunet(x)

        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

class A2AHiFiPlusGenerator(HiFiPlusGenerator):
    def __init__(
        self,
        hifi_resblock="1",
        hifi_upsample_rates=(8, 8, 2, 2),
        hifi_upsample_kernel_sizes=(16, 16, 4, 4),
        hifi_upsample_initial_channel=128,
        hifi_resblock_kernel_sizes=(3, 7, 11),
        hifi_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        hifi_input_channels=128,
        hifi_conv_pre_kernel_size=1,

        upsample_init_channels = 1,
        upsample_block_rates=[2, 2],
        upsample_block_kernel_sizes=[4, 4], 


        residual_channels=64,
        bsft_channels=64,
        nwstack1_blocks=1,
        nwstack2_blocks=1,

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,


        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,

        waveunet_input: Literal["waveform", "hifi", "both"] = "both",
    ):
        super().__init__(
            hifi_resblock=hifi_resblock,
            hifi_upsample_rates=hifi_upsample_rates,
            hifi_upsample_kernel_sizes=hifi_upsample_kernel_sizes,
            hifi_upsample_initial_channel=hifi_upsample_initial_channel,
            hifi_resblock_kernel_sizes=hifi_resblock_kernel_sizes,
            hifi_resblock_dilation_sizes=hifi_resblock_dilation_sizes,
            hifi_input_channels=hifi_input_channels,
            hifi_conv_pre_kernel_size=hifi_conv_pre_kernel_size,

            upsample_init_channels=upsample_init_channels,
            upsample_block_rates=upsample_block_rates,
            upsample_block_kernel_sizes=upsample_block_kernel_sizes,

            residual_channels=residual_channels,
            bsft_channels=bsft_channels,
            nwstack1_blocks=nwstack1_blocks,
            nwstack2_blocks=nwstack2_blocks,

            use_spectralunet=use_spectralunet,
            spectralunet_block_widths=spectralunet_block_widths,
            spectralunet_block_depth=spectralunet_block_depth,
            spectralunet_positional_encoding=spectralunet_positional_encoding,
            kernel_sizes_mrf=None,
            dilations_mrf=None,

            use_waveunet=use_waveunet,
            waveunet_block_widths=waveunet_block_widths,
            waveunet_block_depth=waveunet_block_depth,


            norm_type=norm_type,
            use_skip_connect=use_skip_connect,
        )

        self.waveunet_input = waveunet_input

        self.waveunet_conv_pre = None
        if self.waveunet_input == "waveform":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1, self.hifi.out_channels, 1
                )
            )
        elif self.waveunet_input == "both":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1 + self.hifi.out_channels, self.hifi.out_channels, 1
                )
            )



        self.dim = 513
        self.num_layers = 8
        self.adanorm_num_embeddings = None
        self.intermediate_dim = 1536
        layer_scale_init_value = 1 / self.num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.convnext2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(1)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.final_layer_norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        self.convnext.apply(self._init_weights)
        self.convnext2.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
    
    

    @staticmethod
    def get_stft(x, sampling_rate):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        complex_spec = stft(x, n_fft=1024, num_mels=80, sampling_rate=sampling_rate, hop_size=256, win_size=1024)
        return complex_spec
    

    
    def apply_waveunet_a2a(self, x, x_reference):
        if self.waveunet_input == "waveform":
            x_a = self.waveunet_conv_pre(x_reference)
        elif self.waveunet_input == "both":
            x_a = torch.cat([x, x_reference], 1)
            x_a = self.waveunet_conv_pre(x_a)
        elif self.waveunet_input == "hifi":
            x_a = x
        else:
            raise ValueError
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x

    def forward(self, x, initial_sr, target_sr, **batch):
        initial_x = x.clone()
        batch_size = x.shape[0]
        current_size = initial_x.shape[-1]
        target_size = (target_sr // initial_sr) * current_size
        closest_size = ((current_size + 1023) // 1024) * 1024
        pad_size =  closest_size - current_size
        padded_x = torch.nn.functional.pad(initial_x, (0, pad_size))
        expected_reference_len = (closest_size * target_sr) // initial_sr
        x_reference = batch['reference_wav']
    
        current_reference_len = x_reference.shape[-1]
        pad_reference_len = expected_reference_len - current_reference_len

        padded_reference = torch.nn.functional.pad(x_reference, (0, pad_reference_len)).to(x.device)

        ratio = get_sr_ratio(initial_sr, target_sr)
        num_blocks = get_num_blocks(initial_sr, target_sr)

        if ratio % 3 != 0:
            upsampled_x = self.upsampling_block1(padded_x)
        else:
            upsampled_x=self.upsampling_block_x3(padded_x)
        
        if num_blocks == 1:
            
            band_mask = create_band_mask(initial_sr, target_sr, batch_size, upsampled_x.device)
            x_res = self.nw_stack1(upsampled_x, padded_reference, band_mask)
            
        elif num_blocks == 2:
            if ratio % 3 != 0:
                intermediate_sr = get_intermediate_sr(initial_sr, target_sr)
            else:
                intermediate_sr = initial_sr * 3
            resampled_intermediate = []
            for i in range(batch_size):
                x_single = padded_x[i].cpu().numpy()
                x_resampled_intermediate = librosa.resample(
                    x_single, orig_sr=initial_sr, target_sr=intermediate_sr, res_type="polyphase"
                )
                target_length_intermediate = x_single.shape[-1] * (intermediate_sr // initial_sr)
                if len(x_resampled_intermediate) > target_length_intermediate:
                    x_resampled_intermediate = x_resampled_intermediate[:target_length_intermediate]
                resampled_intermediate.append(x_resampled_intermediate)
            
            x_intermediate_reference = np.stack(resampled_intermediate)
            x_intermediate_reference = torch.tensor(x_intermediate_reference, dtype=padded_x.dtype).to(x.device)
            
            expected_intermediate_len = (closest_size * intermediate_sr) // initial_sr
            current_intermediate_len = x_intermediate_reference.shape[-1]
            pad_intermediate_len = expected_intermediate_len - current_intermediate_len
            padded_intermediate_reference = torch.nn.functional.pad(
                x_intermediate_reference, (0, pad_intermediate_len)
            ).to(x.device)
            
            band_mask_1 = create_band_mask(initial_sr, intermediate_sr, batch_size, upsampled_x.device)
            x_intermediate = self.nw_stack1(upsampled_x, padded_intermediate_reference, band_mask_1)
            
            upsampled_x_intermediate = self.upsampling_block2(x_intermediate)
            
            band_mask_2 = create_band_mask(intermediate_sr, target_sr, batch_size, upsampled_x.device)
            x_res = self.nw_stack2(upsampled_x_intermediate, padded_reference, band_mask_2)
        else:
            raise ValueError(f"Unsupported number of blocks: {num_blocks}. Only 1 or 2 blocks are supported.")


        complex_spec = self.get_stft(x_res, sampling_rate=target_sr)

        real = complex_spec.real   # (B, F, T)
        imag = complex_spec.imag   # (B, F, T)

        for conv_block in self.convnext:
            real = conv_block(real, cond_embedding_id=None)
        real = self.final_layer_norm(real.transpose(1, 2)).transpose(1, 2)

        for conv_block in self.convnext2:
            imag = conv_block(imag, cond_embedding_id=None)
        imag = self.final_layer_norm2(imag.transpose(1, 2)).transpose(1, 2)

        spec = torch.complex(real, imag)

        audio = torch.istft(
            spec,
            1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024).to(x_res.device),
            center=True,
        )
        x_res = audio.unsqueeze(1)
        return x_res[..., :target_size]
    