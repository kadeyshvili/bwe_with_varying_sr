import random
from src.datasets.base_dataset import BaseDataset
import os
import librosa
import torch
from librosa.util import normalize

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
            audios_hr = [audio[:, audio_start : audio_start + segment_size * (hr // lr)] for audio in audios_hr]
        else:
            audios_lr = [torch.nn.functional.pad(audio,(0, segment_size - audio.size(1)),"constant",) for audio in audios_lr]
            audios_hr = [torch.nn.functional.pad(audio,(0, (hr // lr) * segment_size - audio.size(1)),"constant",) for audio in audios_hr]
    audios_lr = [audio.squeeze(0).numpy() for audio in audios_lr]
    audios_hr = [audio.squeeze(0).numpy() for audio in audios_hr]
    return audios_lr, audios_hr

class VCTKDataset(BaseDataset):
    def __init__(
        self,
        dataset_split_file,
        vctk_wavs_dir_lr,
        vctk_wavs_dir_hr,
        segment_size=8192,
        initial_sr=2000,
        target_sr = 4000,
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
        

        input_audio_lr = normalize(vctk_audio_lr)[None] * 0.95
        
        reference_wav = librosa.resample(
                    vctk_audio_lr, orig_sr=self.initial_sr, target_sr=self.target_sr, res_type="polyphase"
                )
        reference_wav = torch.FloatTensor(reference_wav)

        input_audio_hr = normalize(vctk_audio_hr)[None] * 0.95
        assert input_audio_lr.shape[1] == vctk_audio_lr.size
        assert input_audio_hr.shape[1] == vctk_audio_hr.size

        input_audio_lr = torch.FloatTensor(input_audio_lr)
        input_audio_hr = torch.FloatTensor(input_audio_hr)
        melspec_lr = self.mel_creator_lr(input_audio_lr.detach()).squeeze(0)
        melspec_hr = self.mel_creator_hr(input_audio_hr.detach()).squeeze(0)

        return {"wav_lr": input_audio_lr, 'wav_hr': input_audio_hr, 'path_lr' : vctk_fn_lr, 'path_hr':vctk_fn_hr, \
                 'melspec_lr' : melspec_lr, 'melspec_hr' : melspec_hr, 'mode':self.mode, 'reference_wav':reference_wav}

    def __len__(self):
        return len(self.audio_files_lr)


