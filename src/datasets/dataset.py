import os
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from src.model.melspec import  MelSpectrogram

def get_dataset_filelist(dataset_split_file, input_wavs_dir):
    with open(dataset_split_file, "r", encoding="utf-8") as fi:
        files = [os.path.join(input_wavs_dir, fn) for fn in fi.read().split("\n") if len(fn) > 0]
    return files

def split_audios(audios_lr, audios_hr, segment_size, split, lr, hr):
    audios_lr = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios_lr]
    audios_hr = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios_hr]
    if split:
        if audios_lr[0].size(1) >= segment_size:
            max_audio_start = audios_lr[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios_lr = [audio[:, audio_start : audio_start + segment_size] for audio in audios_lr]
            audios_hr = [audio[:, audio_start*(hr // lr) : audio_start*(hr // lr) + segment_size * (hr // lr)] for audio in audios_hr]
        else:
            audios_lr = [torch.nn.functional.pad(audio,(0, segment_size - audio.size(1)),"constant",) for audio in audios_lr]
            audios_hr = [torch.nn.functional.pad(audio,(0, (hr // lr) * segment_size - audio.size(1)),"constant",) for audio in audios_hr]
    audios_lr = [audio.squeeze(0).numpy() for audio in audios_lr]
    audios_hr = [audio.squeeze(0).numpy() for audio in audios_hr]
    return audios_lr, audios_hr

class VCTKDataset(Dataset):
    def __init__(
        self,
        dataset_split_file,
        wavs_dir_4khz=None,
        wavs_dir_8khz=None,
        wavs_dir_16khz=None,
        wavs_dir_24khz=None,
        wavs_dir_48khz=None,
        segment_size=8192,
        split=True,
        mode='train',
        device=None,
    ):
        if wavs_dir_4khz is not None: 
            self.audio_files_4k = get_dataset_filelist(dataset_split_file, wavs_dir_4khz)
        else:
            self.audio_files_4k  = None

        if wavs_dir_8khz is not None: 
            self.audio_files_8k = get_dataset_filelist(dataset_split_file, wavs_dir_8khz)
        else:
            self.audio_files_8k  = None

        if wavs_dir_16khz is not None: 
            self.audio_files_16k = get_dataset_filelist(dataset_split_file, wavs_dir_16khz)
        else:
            self.audio_files_16k  = None

        if wavs_dir_24khz is not None:
            self.audio_files_24k = get_dataset_filelist(dataset_split_file, wavs_dir_24khz)
        else:
            self.audio_files_24k  = None

        if wavs_dir_48khz is not None:
            self.audio_files_48k = get_dataset_filelist(dataset_split_file, wavs_dir_48khz)
        else:
            self.audio_files_48k  = None
        
        
        random.seed(1234)
        self.mode = mode
        self.segment_size = segment_size
        self.split = split
        self.device = device
        if self.audio_files_8k is not None and self.audio_files_16k is not None:
            self.audio_files_lr = self.audio_files_8k
            self.audio_files_hr = self.audio_files_16k
            self.current_mode = '8_16'

        elif self.audio_files_8k is not None and self.audio_files_24k is not None:
            self.audio_files_lr = self.audio_files_8k
            self.audio_files_hr = self.audio_files_24k
            self.current_mode = '8_24'


        elif self.audio_files_4k is not None and self.audio_files_8k is not None:
            self.audio_files_lr = self.audio_files_4k
            self.audio_files_hr = self.audio_files_8k
            self.current_mode = '4_8'


        elif self.audio_files_4k is not None and self.audio_files_16k is not None:
            self.audio_files_lr = self.audio_files_4k
            self.audio_files_hr = self.audio_files_16k
            self.current_mode = '4_16'
            
        elif self.audio_files_4k is not None and self.audio_files_24k is not None:
            self.audio_files_lr = self.audio_files_4k
            self.audio_files_hr = self.audio_files_24k
            self.current_mode = '8_24'

        elif self.audio_files_8k is not None and self.audio_files_48k is not None:
            self.audio_files_lr = self.audio_files_8k
            self.audio_files_hr = self.audio_files_48k
            self.current_mode = '8_48'

        
        self.mel_creator_4k = MelSpectrogram(sr=4000) if wavs_dir_4khz is not None else None
        self.mel_creator_8k = MelSpectrogram(sr=8000) if wavs_dir_8khz is not None else None
        self.mel_creator_16k = MelSpectrogram(sr=16000) if wavs_dir_16khz is not None else None
        self.mel_creator_24k = MelSpectrogram(sr=24000) if wavs_dir_24khz is not None else None
        self.mel_creator_48k = MelSpectrogram(sr=48000) if wavs_dir_48khz is not None else None


        if self.current_mode is not None:
            self.set_batch_mode(self.current_mode)

        
    def set_batch_mode(self, mode=None):
        if mode is not None:
            self.current_mode = mode
        else:
            available_modes = self._get_available_modes()
            if not available_modes:
                raise ValueError("No valid modes available. Please provide at least one pair of sample rate directories.")
            self.current_mode = random.choice(available_modes)

        available_modes = self._get_available_modes()
        if self.current_mode not in available_modes:
            raise ValueError(
                f"Requested mode '{self.current_mode}' is not available. "
                f"Available modes: {available_modes}. "
                f"Please provide the corresponding wavs_dir parameters."
            )
        
        parts = self.current_mode.split('_')
        if len(parts) == 2:
            initial_sr = int(parts[0]) * 1000
            target_sr = int(parts[1]) * 1000
        else:
            raise ValueError(f"Invalid mode format: {self.current_mode}. Expected format: 'initial_target' (e.g., '4_8')")
        
        sr_to_files = {}
        
        if self.audio_files_4k is not None and self.mel_creator_4k is not None:
            sr_to_files[4000] = (self.audio_files_4k, self.mel_creator_4k)
        if self.audio_files_8k is not None and self.mel_creator_8k is not None:
            sr_to_files[8000] = (self.audio_files_8k, self.mel_creator_8k)
        if self.audio_files_16k is not None and self.mel_creator_16k is not None:
            sr_to_files[16000] = (self.audio_files_16k, self.mel_creator_16k)
        if self.audio_files_24k is not None and self.mel_creator_24k is not None:
            sr_to_files[24000] = (self.audio_files_24k, self.mel_creator_24k)
        if self.audio_files_48k is not None and self.mel_creator_48k is not None:
            sr_to_files[48000] = (self.audio_files_48k, self.mel_creator_48k)
        
        if initial_sr not in sr_to_files:
            available_srs = list(sr_to_files.keys())
            raise ValueError(
                f"Unsupported initial_sr: {initial_sr}. Available sample rates: {available_srs}. "
                f"Please provide the corresponding wavs_dir parameter."
            )
        if target_sr not in sr_to_files:
            available_srs = list(sr_to_files.keys())
            raise ValueError(
                f"Unsupported target_sr: {target_sr}. Available sample rates: {available_srs}. "
                f"Please provide the corresponding wavs_dir parameter."
            )
        
        self.audio_files_lr, self.mel_creator_lr = sr_to_files[initial_sr]
        self.audio_files_hr, self.mel_creator_hr = sr_to_files[target_sr]
        self.initial_sr = initial_sr
        self.target_sr = target_sr
        
        return self.current_mode
    
    def _get_available_modes(self):
        available_modes = []
        if self.audio_files_4k is not None and self.audio_files_8k is not None:
            available_modes.append("4_8")
        if self.audio_files_8k is not None and self.audio_files_16k is not None:
            available_modes.append("8_16")
        if self.audio_files_4k is not None and self.audio_files_16k is not None:
            available_modes.append("4_16")
        if self.audio_files_8k is not None and self.audio_files_24k is not None:
            available_modes.append("8_24")
        if self.audio_files_4k is not None and self.audio_files_24k is not None:
            available_modes.append("4_24")
        if self.audio_files_8k is not None and self.audio_files_48k is not None:
            available_modes.append("8_48")
        if self.audio_files_24k is not None and self.audio_files_48k is not None:
            available_modes.append("24_48")
        return available_modes
        
    def __getitem__(self, index_and_mode):
        index, cur_mode = index_and_mode
        self.set_batch_mode(cur_mode)
        vctk_fn_lr = self.audio_files_lr[index]
        vctk_fn_hr = self.audio_files_hr[index]

        vctk_audio_lr = librosa.load(vctk_fn_lr, sr=self.initial_sr, res_type="polyphase",)[0]
        vctk_audio_hr = librosa.load(vctk_fn_hr, sr=self.target_sr, res_type="polyphase",)[0]

        (vctk_audio_lr,), (vctk_audio_hr, ) = split_audios([vctk_audio_lr], [vctk_audio_hr], self.segment_size, self.split, self.initial_sr, self.target_sr)

        peak = max(np.abs(vctk_audio_lr).max(), np.abs(vctk_audio_hr).max())
        scale = 0.95 / peak if peak > 0 else 1.0
        input_audio_lr = (vctk_audio_lr * scale)[None]
        input_audio_hr = (vctk_audio_hr * scale)[None]
        assert input_audio_lr.shape[1] == vctk_audio_lr.size
        assert input_audio_hr.shape[1] == vctk_audio_hr.size

        reference_wav = librosa.resample(
                vctk_audio_lr * scale, orig_sr=self.initial_sr, target_sr=self.target_sr, res_type="polyphase"
            )
        reference_wav = torch.FloatTensor(reference_wav)

        input_audio_lr = torch.FloatTensor(input_audio_lr)
        input_audio_hr = torch.FloatTensor(input_audio_hr)
        melspec_lr = self.mel_creator_lr(input_audio_lr.detach()).squeeze(0)
        melspec_hr = self.mel_creator_hr(input_audio_hr.detach()).squeeze(0)

        return {
            "wav_lr": input_audio_lr, 
            "wav_hr": input_audio_hr, 
            "path_lr": vctk_fn_lr, 
            "path_hr": vctk_fn_hr,
            "melspec_lr": melspec_lr, 
            "melspec_hr": melspec_hr,
            "initial_sr": self.initial_sr,
            "target_sr": self.target_sr,
            'mode':self.mode, 
            'reference_wav':reference_wav
        }

    def __len__(self):
        if self.audio_files_4k is not None:
            return len(self.audio_files_4k)
        elif self.audio_files_8k is not None:
            return len(self.audio_files_8k)
        elif self.audio_files_16k is not None:
            return len(self.audio_files_16k)
        elif self.audio_files_24k is not None:
            return len(self.audio_files_24k)
        elif self.audio_files_48k is not None:
            return len(self.audio_files_48k)
        else:
            raise ValueError("No audio files directories provided.")


class SRConsistentBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, regimes=None):
        self.dataset = dataset
        self.batch_size = batch_size
        
        if regimes is not None:
            self.regimes = regimes
        else:
            self.regimes = self._get_available_regimes()
        
        available_regimes = self._get_available_regimes()
        invalid_regimes = [r for r in self.regimes if r not in available_regimes]
        if invalid_regimes:
            raise ValueError(
                f"Requested regimes {invalid_regimes} are not available. "
                f"Available regimes: {available_regimes}. "
                f"Please provide the corresponding wavs_dir parameters."
            )
    
    def _get_available_regimes(self):
        available_regimes = []
        
        if self.dataset.audio_files_4k is not None and self.dataset.audio_files_8k is not None:
            available_regimes.append("4_8")
        if self.dataset.audio_files_8k is not None and self.dataset.audio_files_16k is not None:
            available_regimes.append("8_16")
        if self.dataset.audio_files_4k is not None and self.dataset.audio_files_16k is not None:
            available_regimes.append("4_16")
        if self.dataset.audio_files_8k is not None and self.dataset.audio_files_24k is not None:
            available_regimes.append("8_24")
        if self.dataset.audio_files_4k is not None and self.dataset.audio_files_24k is not None:
            available_regimes.append("4_24")
        if self.dataset.audio_files_8k is not None and self.dataset.audio_files_48k is not None:
            available_regimes.append("8_48")
        if self.dataset.audio_files_24k is not None and self.dataset.audio_files_48k is not None:
            available_regimes.append("24_48")
        
        return available_regimes
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
    
        num_regimes = len(self.regimes)
        regime_len = len(indices) // num_regimes
        
        regime_indices = {}
        regime_batches = {}
        
        for i, regime in enumerate(self.regimes):
            start_idx = i * regime_len
            end_idx = start_idx + regime_len if i < num_regimes - 1 else len(indices)
            regime_indices[regime] = indices[start_idx:end_idx]
            regime_batches[regime] = []
            
            for j in range(0, len(regime_indices[regime]), self.batch_size):
                batch = [(regime_indices[regime][k], regime) 
                        for k in range(j, min(j + self.batch_size, len(regime_indices[regime])))]
                regime_batches[regime].append(batch)
        
        interleaved_batches = []
        max_batches = max(len(batches) for batches in regime_batches.values())
        
        for i in range(max_batches):
            for regime in self.regimes:
                if i < len(regime_batches[regime]):
                    interleaved_batches.append(regime_batches[regime][i])
        
        return iter(interleaved_batches)
    
    def __len__(self):
        num_regimes = len(self.regimes)
        regime_len = len(self.dataset) // num_regimes
        
        total = 0
        for i in range(num_regimes):
            if i < num_regimes - 1:
                regime_size = regime_len
            else:
                regime_size = len(self.dataset) - (num_regimes - 1) * regime_len
            total += (regime_size + self.batch_size - 1) // self.batch_size
        
        return total