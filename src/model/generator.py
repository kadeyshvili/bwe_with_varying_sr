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


mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        residual = x  # (B, C, F, T)

        x = self.dwconv(x)

        # LayerNorm needs (B, F, T, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)
        x = self.grn(x)

        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)

        return residual + x

def stft(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False,):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)
    if fmin is None:
        fmin=0.0
    if fmax is None:
        fmax = float(sampling_rate) / 2

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    freq_and_time  = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return freq_and_time

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


        self.vocoder_dim = 128
        self.vocoder_num_layers = 8
        self.vocoder_intermediate_dim = 4 * self.vocoder_dim
        self.vocoder_n_fft = 1024
        self.vocoder_hop_size = 256
        self.vocoder_win_size = 1024

        self.mag_input_conv = nn.Conv2d(1, self.vocoder_dim, kernel_size=3, padding=1)

        self.mag_blocks = nn.ModuleList(
            [
                ConvNeXtBlock2D(self.vocoder_dim, self.vocoder_intermediate_dim)
                for _ in range(self.vocoder_num_layers)
            ]
        )

        self.mag_output_conv = nn.Conv2d(self.vocoder_dim, 1, kernel_size=3, padding=1)

        self.phase_input_conv = nn.Conv2d(1, self.vocoder_dim, kernel_size=3, padding=1)

        self.phase_blocks = nn.ModuleList(
            [
                ConvNeXtBlock2D(self.vocoder_dim, self.vocoder_intermediate_dim)
                for _ in range(self.vocoder_num_layers)
            ]
        )

        self.phase_output_conv = nn.Conv2d(self.vocoder_dim, 1, kernel_size=3, padding=1)

        for m in [self.mag_input_conv, self.mag_output_conv,
                  self.phase_input_conv, self.phase_output_conv,
                  *self.mag_blocks, *self.phase_blocks]:
            m.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    
    

    @staticmethod
    def get_stft(x, sampling_rate):
        # x: (B, 1, T) -> squeeze -> (B, T) -> stft -> (B, 513, T_frames)
        x = x.squeeze(1)
        x = stft(x, n_fft=1024, num_mels=80, sampling_rate=sampling_rate, hop_size=256, win_size=1024, fmin=None, fmax=None)
        return x
    
    
    
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


        spec = self.get_stft(x_res, sampling_rate=target_sr)

        mag = torch.abs(spec)
        phase = torch.angle(spec)

        mag = mag.unsqueeze(1)
        phase = phase.unsqueeze(1)

        mag_feat = self.mag_input_conv(mag)
        for block in self.mag_blocks:
            mag_feat = block(mag_feat)
        mag_out = self.mag_output_conv(mag_feat)

        phase_feat = self.phase_input_conv(phase)
        for block in self.phase_blocks:
            phase_feat = block(phase_feat)
        phase_out = self.phase_output_conv(phase_feat)

        mag_pred = mag + mag_out
        phase_pred = phase + phase_out

        real_pred = mag_pred * torch.cos(phase_pred)
        imag_pred = mag_pred * torch.sin(phase_pred)

        complex_spec = torch.complex(
            real_pred.squeeze(1),
            imag_pred.squeeze(1)
        )

        audio = torch.istft(
            complex_spec,
            self.vocoder_n_fft,
            hop_length=self.vocoder_hop_size,
            win_length=self.vocoder_win_size,
            window=torch.hann_window(self.vocoder_win_size).to(spec.device),
            center=True,
        )
        x_res = audio.unsqueeze(1)

        return x_res[..., :target_size]
    