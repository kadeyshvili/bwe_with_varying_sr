import os
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from src.model.melspec import  MelSpectrogram
from librosa.util import normalize

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
            audios_hr = [audio[:, audio_start : audio_start + segment_size * (hr // lr)] for audio in audios_hr]
        else:
            audios_lr = [torch.nn.functional.pad(audio,(0, segment_size - audio.size(1)),"constant",) for audio in audios_lr]
            audios_hr = [torch.nn.functional.pad(audio,(0, (hr // lr) * segment_size - audio.size(1)),"constant",) for audio in audios_hr]
    audios_lr = [audio.squeeze(0).numpy() for audio in audios_lr]
    audios_hr = [audio.squeeze(0).numpy() for audio in audios_hr]
    return audios_lr, audios_hr

class VCTKDatasetSeparate(Dataset):
    def __init__(
        self,
        dataset_split_file,
        wavs_dir_4khz,
        wavs_dir_8khz,
        wavs_dir_16khz,
        segment_size=8192,
        split=True,
        mode='train',
        device=None,
    ):
        self.audio_files_4k = get_dataset_filelist(dataset_split_file, wavs_dir_4khz)
        self.audio_files_8k = get_dataset_filelist(dataset_split_file, wavs_dir_8khz)
        self.audio_files_16k = get_dataset_filelist(dataset_split_file, wavs_dir_16khz)
        
        random.seed(1234)
        self.mode = mode
        self.segment_size = segment_size
        self.split = split
        self.device = device
        
        self.current_mode = "8_16"  
        
        self.mel_creator_4k = MelSpectrogram(sr=4000)
        self.mel_creator_8k = MelSpectrogram(sr=8000)
        self.mel_creator_16k = MelSpectrogram(sr=16000)

        self.set_batch_mode(self.current_mode)
        
    def set_batch_mode(self, mode=None):
        if mode is not None:
            self.current_mode = mode
        else:
            mode_choice = random.random()
            if mode_choice < 0.5:
                self.current_mode = "4_8"
            else:
                self.current_mode = "8_16"
                
        if self.current_mode == "4_8":
            self.audio_files_lr = self.audio_files_4k
            self.audio_files_hr = self.audio_files_8k
            self.initial_sr = 4000
            self.target_sr = 8000
            self.mel_creator_lr = self.mel_creator_4k
            self.mel_creator_hr = self.mel_creator_8k
        elif self.current_mode == "8_16": 
            self.audio_files_lr = self.audio_files_8k
            self.audio_files_hr = self.audio_files_16k
            self.initial_sr = 8000
            self.target_sr = 16000
            self.mel_creator_lr = self.mel_creator_8k
            self.mel_creator_hr = self.mel_creator_16k
        return self.current_mode
        
    def __getitem__(self, index_and_mode):
        index, cur_mode = index_and_mode
        self.set_batch_mode(cur_mode)
        vctk_fn_lr = self.audio_files_lr[index]
        vctk_fn_hr = self.audio_files_hr[index]

        vctk_audio_lr = librosa.load(vctk_fn_lr, sr=self.initial_sr, res_type="polyphase",)[0]
        vctk_audio_hr = librosa.load(vctk_fn_hr, sr=self.target_sr, res_type="polyphase",)[0]

        (vctk_audio_lr,), (vctk_audio_hr, ) = split_audios([vctk_audio_lr], [vctk_audio_hr], self.segment_size, self.split, self.initial_sr, self.target_sr)

        input_audio_lr = normalize(vctk_audio_lr)[None] * 0.95
        input_audio_hr = normalize(vctk_audio_hr)[None] * 0.95
        assert input_audio_lr.shape[1] == vctk_audio_lr.size
        assert input_audio_hr.shape[1] == vctk_audio_hr.size

        reference_wav = librosa.resample(
                vctk_audio_lr, orig_sr=self.initial_sr, target_sr=self.target_sr, res_type="polyphase"
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
            "mode": self.current_mode,
            "initial_sr": self.initial_sr,
            "target_sr": self.target_sr,
            'mode':self.mode, 
            'reference_wav':reference_wav
        }

    def __len__(self):
        return len(self.audio_files_lr)


class SRConsistentBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
    
        half_len = len(indices) // 2
        indices_4_8 = indices[:half_len]
        indices_8_16 = indices[half_len:]
        
        mode_4_8_batches = []
        mode_8_16_batches = []
        for i in range(0, len(indices_4_8), self.batch_size):
            batch = [(indices_4_8[j], '4_8') for j in range(i, min(i + self.batch_size, len(indices_4_8)))]
            mode_4_8_batches.append(batch)
        
        for i in range(0, len(indices_8_16), self.batch_size):
            batch = [(indices_8_16[j], '8_16') for j in range(i, min(i + self.batch_size, len(indices_8_16)))]
            mode_8_16_batches.append(batch)
        
        interleaved_batches = []
        max_batches = max(len(mode_4_8_batches), len(mode_8_16_batches))
        
        for i in range(max_batches):
            if i < len(mode_4_8_batches):
                interleaved_batches.append(mode_4_8_batches[i])
            if i < len(mode_8_16_batches):
                interleaved_batches.append(mode_8_16_batches[i])
        
        return iter(interleaved_batches)
    
    def __len__(self):
        total_4_8 = (len(self.dataset) // 2 + self.batch_size - 1) // self.batch_size
        total_8_16 = (len(self.dataset) - len(self.dataset) // 2 + self.batch_size - 1) // self.batch_size
        return total_4_8 + total_8_16