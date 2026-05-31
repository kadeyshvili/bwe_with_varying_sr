import random
from src.datasets.base_dataset import BaseDataset
import os
import librosa
import torch
import numpy as np
from .dataset import split_audios, get_dataset_filelist
from src.model.melspec import  MelSpectrogram


class VCTKTestDataset(BaseDataset):
    def __init__(
        self,
        dataset_split_file,
        vctk_wavs_dir_lr,
        vctk_wavs_dir_hr,
        segment_size=8192,
        initial_sr=4000,
        target_sr = 16000,
        mode='train',
        split=True,
        device=None,
    ):
        self.audio_files_lr = get_dataset_filelist(dataset_split_file,
                                                vctk_wavs_dir_lr)
        self.audio_files_hr = get_dataset_filelist(dataset_split_file,
                                                vctk_wavs_dir_hr)
        random.seed(1234)
        self.mode = mode
        self.segment_size = segment_size
        self.initial_sr = initial_sr
        self.split = split
        self.device = device
        self.target_sr = target_sr
        self.mel_creator_lr = MelSpectrogram(sr=initial_sr)
        self.mel_creator_hr = MelSpectrogram(sr=target_sr)

    def __getitem__(self, index):
        vctk_fn_lr = self.audio_files_lr[index]
        vctk_fn_hr = self.audio_files_hr[index]

        vctk_audio_lr = librosa.load(vctk_fn_lr, sr=self.initial_sr, res_type="polyphase",)[0]
        vctk_audio_hr = librosa.load(vctk_fn_hr, sr=self.target_sr, res_type="polyphase",)[0]

        (vctk_audio_lr,), (vctk_audio_hr, ) = split_audios([vctk_audio_lr], [vctk_audio_hr], self.segment_size, self.split, self.initial_sr, self.target_sr)
        

        peak = max(np.abs(vctk_audio_lr).max(), np.abs(vctk_audio_hr).max())
        scale = 0.95 / peak if peak > 0 else 1.0
        input_audio_lr = (vctk_audio_lr * scale)[None]

        reference_wav = librosa.resample(
                    vctk_audio_lr * scale, orig_sr=self.initial_sr, target_sr=self.target_sr, res_type="polyphase"
                )
        reference_wav = torch.FloatTensor(reference_wav)

        input_audio_hr = (vctk_audio_hr * scale)[None]
        assert input_audio_lr.shape[1] == vctk_audio_lr.size
        assert input_audio_hr.shape[1] == vctk_audio_hr.size

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
        return len(self.audio_files_lr)

