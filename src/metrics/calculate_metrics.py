import itertools
from abc import ABC, abstractmethod
import os
import urllib.request
import librosa
import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn

# import log_utils
from src.metrics.metric_denoising import composite_eval
from src.metrics.metric_nets import Wav2Vec2MOS


class Metric(ABC):
    name = "Abstract Metric"

    def __init__(self, num_splits=5, big_val_size=500, name=None):
        self.num_splits = num_splits

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.duration = None  # calculated in Metric.compute()
        self.val_size = None
        self.result = defaultdict(list)
        self.big_val_size = big_val_size

    @abstractmethod
    def better(self, first, second):
        pass

    @abstractmethod
    def __call__(self, samples, real_samples, epoch_num, epoch_info):
        pass

    def compute(self, samples, real_samples, epoch_num, epoch_info):
        # with log_utils.Timer() as timer:
        self.__call__(samples, real_samples, epoch_num, epoch_info)
        # self.result["dur"] = timer.duration
        self.result["val_size"] = samples.shape[0]

        if "best_mean" not in self.result or self.better(
            self.result["mean"], self.result["best_mean"]
        ):
            self.result["best_mean"] = self.result["mean"]
            self.result["best_std"] = self.result["std"]
            self.result["best_epoch"] = epoch_num

        if self.result["val_size"] >= 200:  # for now
            self.result["big_val_mean"] = self.result["mean"]
            if "best_big_val_mean" not in self.result or self.better(
                self.result["big_val_mean"], self.result["best_big_val_mean"]
            ):
                self.result["best_big_val_mean"] = self.result["big_val_mean"]

    def save_result(self, epoch_info):
        metric_name = self.name
        for key, value in self.result.items():
            epoch_info[f"metrics_{key}/{metric_name}"] = value


class MOSNet(Metric):
    name = "MOS"

    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        path = "src/weights/wave2vec2mos.pth"

        if (not os.path.exists(path)):
            print("Downloading the checkpoint for WV-MOS")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(
                "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
                path
            )
            print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))

        self.mos_net = Wav2Vec2MOS(path)
        self.sr = sr

    def better(self, first, second):
        return first > second

    def _compute_per_split(self, split):
        return self.mos_net.calculate(split)

    def __call__(self, samples, real_samples):
        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=required_sr
        ).to(self.device)

        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]


        splits = [
            samples[i : i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]
        self.result["mean"].append(np.mean(fid_per_splits))
        self.result["std"].append(np.std(fid_per_splits))


class ScaleInvariantSignalToDistortionRatio(Metric):
    """
    See https://arxiv.org/pdf/1811.02508.pdf
    """

    name = "SISDR"

    def better(self, first, second):
        return first > second

    def __call__(self, samples, real_samples):
        real_samples, samples = real_samples.squeeze(), samples.squeeze()
        if real_samples.dim() == 1:
            real_samples = real_samples[None]
            samples = samples[None]
        alpha = (samples * real_samples).sum(
            dim=1, keepdim=True
        ) / real_samples.square().sum(dim=1, keepdim=True)
        real_samples_scaled = alpha * real_samples
        e_target = real_samples_scaled.square().sum(dim=1)
        e_res = (samples - real_samples_scaled).square().sum(dim=1)
        si_sdr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        self.result["mean"].append(np.mean(si_sdr))
        self.result["std"].append(np.std(si_sdr))


class SignalToNoiseRatio(Metric):

    name = "SNR"

    def better(self, first, second):
        return first > second

    def __call__(self, samples, real_samples):
        if real_samples.dim() == 3:
            real_samples = real_samples.squeeze(1)
        if samples.dim() == 3:
            samples = samples.squeeze(1)
        
        if real_samples.dim() == 1:
            real_samples = real_samples.unsqueeze(0)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)
 
        e_target = real_samples.square().sum(dim=1)
        e_res = (samples - real_samples).square().sum(dim=1)
        snr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        self.result["mean"].append(np.mean(snr))
        self.result["std"].append(np.std(snr))


class VGGDistance(Metric):

    name = "VGG_dist"

    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.model.eval()

    def better(self, first, second):
        return first < second

    def __call__(self, samples, real_samples):
        if real_samples.dim() == 3:
            real_samples = real_samples.squeeze(1)
        if samples.dim() == 3:
            samples = samples.squeeze(1)
        
        real_samples = real_samples.cpu().numpy()
        samples = samples.cpu().numpy()
        
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
        if samples.ndim == 1:
            samples = samples[None]
        
        assert (
            samples.shape[1] >= 16384
        ), f"too small segment size {samples.shape[1]}, need at least 16384 samples"
        
        real_embs, fake_embs = [], []
        for real_s, fake_s in zip(real_samples, samples):
            real_embs.append(self.model(real_s, self.sr))
            fake_embs.append(self.model(fake_s, self.sr))
        real_embs = torch.stack(real_embs, dim=0)
        fake_embs = torch.stack(fake_embs, dim=0)
        dist = (real_embs - fake_embs).square().mean(dim=1)
        dist = dist.cpu().detach().numpy()

        self.result["mean"].append(np.mean(dist))
        self.result["std"].append(np.std(dist))



class STFTMag(nn.Module):
    def __init__(self, nfft=1024, hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        stft = torch.stft(x, self.nfft, self.hop, window=self.window, return_complex=False)
        mag = torch.norm(stft, p=2, dim=-1)
        return mag


class LSD(Metric):
    name = "LSD"

    def better(self, first, second):
        return first < second
    def __call__(self, out_sig, ref_sig):
        """
        Compute LSD (log spectral distance)
        Arguments:
            out_sig: vector (torch.Tensor), enhanced signal [B,T]
            ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
        """

        out_sig = out_sig.squeeze(1).cpu()
        ref_sig = ref_sig.squeeze(1).cpu()

        stft = STFTMag(2048, 512)
        batch_size = out_sig.shape[0]
        lsd_list = []
        
        for i in range(batch_size):
            sp = torch.log10(stft(ref_sig[i]).square().clamp(1e-8))
            st = torch.log10(stft(out_sig[i]).square().clamp(1e-8))
            lsd_sample = (sp - st).square().mean(dim=1).sqrt().mean()
            lsd_list.append(lsd_sample.item())

        self.result["mean"].append(np.mean(lsd_list))
        self.result["std"].append(np.std(lsd_list))


class LSD_LF(Metric):
    name = "LSD_LF"

    def better(self, first, second):
        return first < second
    def __call__(self, out_sig, ref_sig, target_sr, initial_sr):
        """
        Compute LSD (log spectral distance)
        Arguments:
            out_sig: vector (torch.Tensor), enhanced signal [B,T]
            ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
        """

        out_sig = out_sig.squeeze(1).cpu()
        ref_sig = ref_sig.squeeze(1).cpu()

        stft = STFTMag(2048, 512)
        hf = int(1025 * (initial_sr / target_sr))
        batch_size = out_sig.shape[0]
        lsd_lf_list = []
        
        for i in range(batch_size):
            sp = torch.log10(stft(ref_sig[i]).square().clamp(1e-8))
            st = torch.log10(stft(out_sig[i]).square().clamp(1e-8))
            
            lsd_lf_sample = (sp[:hf, :] - st[:hf, :]).square().mean(dim=1).sqrt().mean()
            lsd_lf_list.append(lsd_lf_sample.item())
 
        self.result["mean"].append(np.mean(lsd_lf_list))
        self.result["std"].append(np.std(lsd_lf_list))



class LSD_HF(Metric):
    name = "LSD_HF"

    def better(self, first, second):
        return first < second
    def __call__(self, out_sig, ref_sig, target_sr, initial_sr):
        """
        Compute LSD (log spectral distance)
        Arguments:
            out_sig: vector (torch.Tensor), enhanced signal [B,T]
            ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
        """

        out_sig = out_sig.squeeze(1).cpu()
        ref_sig = ref_sig.squeeze(1).cpu()

        stft = STFTMag(2048, 512)
        hf = int(1025 * (initial_sr / target_sr))
        batch_size = out_sig.shape[0]
        lsd_hf_list = []
        
        for i in range(batch_size):
            sp = torch.log10(stft(ref_sig[i]).square().clamp(1e-8))
            st = torch.log10(stft(out_sig[i]).square().clamp(1e-8))
            
            lsd_hf_sample = (sp[hf:, :] - st[hf:, :]).square().mean(dim=1).sqrt().mean()
            lsd_hf_list.append(lsd_hf_sample.item())
        self.result["mean"].append(np.mean(lsd_hf_list))
        self.result["std"].append(np.std(lsd_hf_list))


class STOI(Metric):
    name = "STOI"

    def better(self, first, second):
        return first > second

    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr

    def __call__(self, samples, real_samples):
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        stois = []
        for s_real, s_fake in zip(real_samples, samples):
            s = stoi(s_real, s_fake, self.sr, extended=True)
            stois.append(s)
        self.result["mean"].append(np.mean(stois))
        self.result["std"].append(np.std(stois))


class PESQ(Metric):
    name = "PESQ"

    def better(self, first, second):
        return first > second

    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr

    def __call__(self, samples, real_samples):
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]
        real_samples = real_samples.squeeze(1).cpu().numpy()
        samples = samples.squeeze(1).cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        pesqs = []
        for s_real, s_fake in zip(real_samples, samples):
            try:
                p = pesq(self.sr, s_real, s_fake, mode="wb")
            except:
                p = 1
            pesqs.append(p)

        self.result["mean"].append(np.mean(pesqs))
        self.result["std"].append(np.std(pesqs))


class CSEMetric(Metric):
    # sampling rate is 16000
    def better(self, first, second):
        return first > second

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = None

    def __call__(self, samples, real_samples):
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]
        real_samples = real_samples.squeeze(1).cpu().numpy()
        samples = samples.squeeze(1).cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        res = list()
        for s_real, s_fake in zip(real_samples, samples):
            r = self.func(s_real, s_fake)
            res.append(r)

        self.result["mean"].append(np.mean(res))
        self.result["std"].append(np.std(res))


class CSIG(CSEMetric):
    # sampling rate is 16000
    name = "CSIG"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y: composite_eval(x, y)["csig"]


class CBAK(CSEMetric):
    # sampling rate is 16000
    name = "CBAK"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y: composite_eval(x, y)["cbak"]


class COVL(CSEMetric):
    # sampling rate is 16000
    name = "COVL"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y: composite_eval(x, y)["covl"]


def calculate_all_metrics(wavs, reference_wavs, metrics, initial_sr, target_sr):
    scores = {}
    for metric in metrics:
        if metric.name=='LSD_LF' or metric.name=='LSD_HF':
            metric.__call__(wavs, reference_wavs, target_sr, initial_sr)
        else:
            metric.__call__(wavs, reference_wavs)
    for metric in metrics:
        results = metric.result
        mean_val = results['mean']
        std_val = results['std']
        scores[metric.name] = (mean_val, std_val)
    return scores
