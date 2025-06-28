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
import itertools

from src.metrics.metric_denoising import composite_eval
from src.metrics.metric_nets import Wav2Vec2MOS
 
 
class MetricSeparate(ABC):
    name = "Abstract Metric"
 
    def __init__(self, num_splits=5, big_val_size=500, name=None):
        self.num_splits = num_splits
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.duration = None
        self.val_size = None
        self.big_val_size = big_val_size
        self.name = name if name is not None else type(self).__name__
 
        self.results = {
            "4_8": defaultdict(list),
            "8_16": defaultdict(list),
            "default": defaultdict(list)
        }
        self.current_mode = "default"
 
    @abstractmethod
    def better(self, first, second):
        pass
 
    @abstractmethod
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        if initial_sr is not None and target_sr is not None:
            if initial_sr == 4000 and target_sr == 8000:
                self.current_mode = "4_8"
            elif initial_sr == 8000 and target_sr == 16000:
                self.current_mode = "8_16"
            else:
                self.current_mode = "default"
        else:
            self.current_mode = "default"
 
        self.results[self.current_mode] = defaultdict(list)
 
        pass
 
    def get_result(self, mode=None):
        """Получение результатов для указанного режима"""
        if mode is None:
            mode = self.current_mode
        return self.results[mode]
 
    def save_result(self, epoch_info):
        """Сохранение результатов в epoch_info"""
        for mode, results in self.results.items():
            if mode == "default" or not results:
                continue
 
            for key, value in results.items():
                if value:
                    metric_name = f"{self.name}_{mode}"
                    epoch_info[f"val_{metric_name}"] = np.mean(value)
 
 
class MOSNetSeparate(MetricSeparate):
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
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        super().__call__(samples, real_samples, target_sr, initial_sr)

        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=target_sr, new_freq=required_sr
        ).to(self.device)
        samples = samples.to(self.device)
        
        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]

        splits = [
            samples[i : i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]

        self.results[self.current_mode]["mean"].append(np.mean(fid_per_splits))
        self.results[self.current_mode]["std"].append(np.std(fid_per_splits))

        return np.mean(fid_per_splits)
 
 
class ScaleInvariantSignalToDistortionRatioSeparate(MetricSeparate):
    """
    See https://arxiv.org/pdf/1811.02508.pdf
    """
    name = "SISDR"
 
    def better(self, first, second):
        return first > second
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        super().__call__(samples, real_samples, target_sr, initial_sr)
 
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
 
        self.results[self.current_mode]["mean"].append(np.mean(si_sdr))
        self.results[self.current_mode]["std"].append(np.std(si_sdr))
 
        return np.mean(si_sdr)
 
 
class SignalToNoiseRatioSeparate(MetricSeparate):
    name = "SNR"
 
    def better(self, first, second):
        return first > second
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        super().__call__(samples, real_samples, target_sr, initial_sr)
 
        real_samples, samples = real_samples.squeeze(), samples.squeeze()
        if real_samples.dim() == 1:
            real_samples = real_samples[None]
            samples = samples[None]
 
        e_target = real_samples.square().sum(dim=1)
        e_res = (samples - real_samples).square().sum(dim=1)
        snr = 10 * torch.log10(e_target / e_res).cpu().numpy()
 
        self.results[self.current_mode]["mean"].append(np.mean(snr))
        self.results[self.current_mode]["std"].append(np.std(snr))
 
        return np.mean(snr)
 
 
class VGGDistanceSeparate(MetricSeparate):
    name = "VGG_dist"
 
    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.model.eval()
 
    def better(self, first, second):
        return first < second
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        # Определяем режим апсемплинга
        super().__call__(samples, real_samples, target_sr, initial_sr)
 
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        assert (
            samples.shape[1] >= 16384
        ), "too small segment size, everything will fall in this function"
        real_embs, fake_embs = [], []
        for real_s, fake_s in zip(real_samples, samples):
            real_embs.append(self.model(real_s, self.sr))
            fake_embs.append(self.model(fake_s, self.sr))
        real_embs = torch.stack(real_embs, dim=0)
        fake_embs = torch.stack(fake_embs, dim=0)
        dist = (real_embs - fake_embs).square().mean(dim=1)
        dist = dist.cpu().detach().numpy()
 
        self.results[self.current_mode]["mean"].append(np.mean(dist))
        self.results[self.current_mode]["std"].append(np.std(dist))
 
        return np.mean(dist)
 
 
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
 
 
class LSDSeparate(MetricSeparate):
    name = "LSD"
 
    def better(self, first, second):
        return first < second
 
    def __call__(self, out_sig, ref_sig, target_sr, initial_sr=None):
        """
        Compute LSD (log spectral distance)
        Arguments:
            out_sig: vector (torch.Tensor), enhanced signal [B,T]
            ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
        """
        super().__call__(out_sig, ref_sig, target_sr, initial_sr)
 
        out_sig = out_sig.squeeze().cpu()
        ref_sig = ref_sig.squeeze().cpu()
 
        stft = STFTMag(2048, 512)
        sp = torch.log10(stft(ref_sig).square().clamp(1e-8))
        st = torch.log10(stft(out_sig).square().clamp(1e-8))   
        lsd = (sp - st).square().mean(dim=1).sqrt().mean()
        lsd = lsd.cpu().numpy()
 
        self.results[self.current_mode]["mean"].append(np.mean(lsd))
        self.results[self.current_mode]["std"].append(np.std(lsd))
 
        return np.mean(lsd)
 
 
class LSD_LF_Separate(MetricSeparate):
    name = "LSD_LF"
 
    def better(self, first, second):
        return first < second
 
    def __call__(self, out_sig, ref_sig, initial_sr, target_sr):
        """
        Compute LSD (log spectral distance) for low frequencies
        Arguments:
            out_sig: vector (torch.Tensor), enhanced signal [B,T]
            ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
            initial_sr: initial sample rate
            target_sr: target sample rate
        """
        if initial_sr == 4000 and target_sr == 8000:
            self.current_mode = "4_8"
        elif initial_sr == 8000 and target_sr == 16000:
            self.current_mode = "8_16"
        else:
            self.current_mode = "default"
        self.results[self.current_mode] = defaultdict(list)
 
        out_sig = out_sig.squeeze().cpu()
        ref_sig = ref_sig.squeeze().cpu()
 
        stft = STFTMag(2048, 512)
        hf = int(1025 * (initial_sr / target_sr))
        sp = torch.log10(stft(ref_sig).square().clamp(1e-8))
        st = torch.log10(stft(out_sig).square().clamp(1e-8))
        lsd = (sp[:hf, :] - st[:hf, :]).square().mean(dim=1).sqrt().mean()
        lsd = lsd.cpu().numpy()
 
        self.results[self.current_mode]["mean"].append(np.mean(lsd))
        self.results[self.current_mode]["std"].append(np.std(lsd))
 
        return np.mean(lsd)
 
 
class LSD_HF_Separate(MetricSeparate):
    name = "LSD_HF"
 
    def better(self, first, second):
        return first < second
 
    def __call__(self, out_sig, ref_sig, initial_sr, target_sr):
        """
        Compute LSD (log spectral distance) for high frequencies
        Arguments:
            out_sig: vector (torch.Tensor), enhanced signal [B,T]
            ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
            initial_sr: initial sample rate
            target_sr: target sample rate
        """
        if initial_sr == 4000 and target_sr == 8000:
            self.current_mode = "4_8"
        elif initial_sr == 8000 and target_sr == 16000:
            self.current_mode = "8_16"
        else:
            self.current_mode = "default"
        self.results[self.current_mode] = defaultdict(list)
 
        out_sig = out_sig.squeeze().cpu()
        ref_sig = ref_sig.squeeze().cpu()
 
        stft = STFTMag(2048, 512)
        hf = int(1025 * (initial_sr / target_sr))
        sp = torch.log10(stft(ref_sig).square().clamp(1e-8))
        st = torch.log10(stft(out_sig).square().clamp(1e-8))
        lsd_hf = (sp[hf:, :] - st[hf:, :]).square().mean(dim=1).sqrt().mean()
        lsd_hf = lsd_hf.cpu().numpy()
 
        self.results[self.current_mode]["mean"].append(np.mean(lsd_hf))
        self.results[self.current_mode]["std"].append(np.std(lsd_hf))
 
        return np.mean(lsd_hf)
 
 
class STOI_Separate(MetricSeparate):
    name = "STOI"
 
    def better(self, first, second):
        return first > second
 
    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        super().__call__(samples, real_samples, target_sr, initial_sr)
 
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]
 
        stois = []
        for s_real, s_fake in zip(real_samples, samples):
            s = stoi(s_real, s_fake, self.sr, extended=True)
            stois.append(s)
 
        self.results[self.current_mode]["mean"].append(np.mean(stois))
        self.results[self.current_mode]["std"].append(np.std(stois))
 
        return np.mean(stois)
 
 
class PESQ_Separate(MetricSeparate):
    name = "PESQ"
 
    def better(self, first, second):
        return first > second
 
    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        super().__call__(samples, real_samples, target_sr, initial_sr)
 
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
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
 
        self.results[self.current_mode]["mean"].append(np.mean(pesqs))
        self.results[self.current_mode]["std"].append(np.std(pesqs))
 
        return np.mean(pesqs)
 
 
class CSEMetric_Separate(MetricSeparate):
    def better(self, first, second):
        return first > second
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = None
 
    def __call__(self, samples, real_samples, target_sr, initial_sr=None):
        super().__call__(samples, real_samples, target_sr, initial_sr)
 
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]
 
        res = list()
        for s_real, s_fake in zip(real_samples, samples):
            r = self.func(s_real, s_fake, target_sr)
            res.append(r)
 
        self.results[self.current_mode]["mean"].append(np.mean(res))
        self.results[self.current_mode]["std"].append(np.std(res))
 
        return np.mean(res)
 
 
class CSIG_Separate(CSEMetric_Separate):
    name = "CSIG"
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y, target_sr: composite_eval(x, y, target_sr)["csig"]
 
 
class CBAK_Separate(CSEMetric_Separate):
    name = "CBAK"
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y, target_sr: composite_eval(x, y, target_sr)["cbak"]
 
 
class COVL_Separate(CSEMetric_Separate):
    name = "COVL"
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = lambda x, y, target_sr: composite_eval(x, y, target_sr)["covl"]
 
 
def calculate_all_metrics(wavs, reference_wavs, metrics, initial_sr, target_sr, n_max_files=None):
    mode_key = f"{initial_sr // 1000}_{target_sr // 1000}"
    metrics_to_use = []
    for metric in metrics:
        if mode_key in metric.name:
            metrics_to_use.append(metric)
 
    scores = {}
    for x, y in tqdm(
        itertools.islice(zip(wavs, reference_wavs), n_max_files),
        total=n_max_files if n_max_files is not None else len(wavs),
        desc="Calculating metrics",
    ):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        x = x[0, :]
        y = y[0, :]
        x = librosa.util.normalize(x[: min(len(x), len(y))])
        y = librosa.util.normalize(y[: min(len(x), len(y))])
        x = torch.from_numpy(x)[None, None]
        y = torch.from_numpy(y)[None, None]
        for metric in metrics:
            if "LSD_HF" in metric.name or "LSD_LF" in metric.name:
                metric.__call__(x, y, initial_sr, target_sr)
            else:
                metric.__call__(x, y, target_sr, initial_sr)
 
    for metric in metrics_to_use:
        results = metric.get_result(f"{initial_sr // 1000}_{target_sr // 1000}")
 
        if results["mean"]:
            mean_val = np.mean(results["mean"])
            std_val = np.std(results["mean"]) if len(results["mean"]) > 1 else 0
            scores[metric.name] = (mean_val, std_val)
 
    return scores