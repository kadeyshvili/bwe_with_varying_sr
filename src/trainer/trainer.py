from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.model import HiFiGANWithMRF
from hydra.utils import instantiate
import librosa
import numpy as np



class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)

        initial_wav = batch['wav_lr']
        target_wav = batch['wav_hr']
        initial_sr = batch['initial_sr']
        target_sr = batch['target_sr']
        model_instance = instantiate(self.config.model)
        if isinstance(model_instance, HiFiGANWithMRF):
            wav_fake = self.model.generator(initial_wav, initial_sr, target_sr)
        else:
            wav_fake = self.model.generator(initial_wav, **batch)


        if initial_sr==8000 and target_sr==16000:
            resampled_audio_4khz = []
            for i in range(initial_wav.shape[0]):
                x_single = target_wav[i].cpu().numpy()
                x_resampled = librosa.resample(
                    x_single, orig_sr=16000, target_sr=4000, res_type="polyphase"
                )
                
                resampled_audio_4khz.append(x_resampled)
            x_resampled_4khz= np.stack(resampled_audio_4khz)
            x_resampled_4khz = torch.tensor(x_resampled_4khz, dtype=target_wav.dtype).to(target_wav.device)
            batch_clean = {k: v for k, v in batch.items() if k not in ("initial_sr", "target_sr")}
            batch['wav_16k_from_4k_gen'] = self.model.generator(x_resampled_4khz , initial_sr=4000, target_sr=16000, **batch_clean)
 
        if target_wav.shape != wav_fake.shape:
            wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
        batch["generated_wav"] = wav_fake
        if initial_sr==4000 and target_sr==8000:
            mel_spec_fake = self.create_mel_spec_4_8(wav_fake).squeeze(1)
        elif initial_sr==8000 and target_sr == 16000:
            mel_spec_fake = self.create_mel_spec_8_16(wav_fake).squeeze(1)
        batch['mel_spec_fake'] = mel_spec_fake
        if self.is_train:
            self.disc_optimizer.zero_grad()

        mpd_gt_out, _, mpd_fake_out, _ = self.model.mpd(target_wav, wav_fake.detach())

        msd_gt_out, _,  msd_fake_out, _ = self.model.msd(target_wav, wav_fake.detach())
        batch['mpd_gt_out'] = mpd_gt_out
        batch['mpd_fake_out'] = mpd_fake_out
        batch['msd_gt_out'] = msd_gt_out
        batch['msd_fake_out'] = msd_fake_out
        if initial_sr==8000 and target_sr == 16000:
            mpd_disc_loss_8_16, msd_disc_loss_8_16, disc_loss_8_16 = self.criterion.discriminator_loss_8_16(batch)
        elif initial_sr == 4000 and target_sr == 8000:
            mpd_disc_loss_4_8, msd_disc_loss_4_8, disc_loss_4_8 = self.criterion.discriminator_loss_4_8(batch)



        if self.is_train:
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)

        if self.is_train:
            if initial_sr==8000 and target_sr == 16000:
                disc_loss_8_16.backward()
            elif initial_sr == 4000 and target_sr == 8000:
                disc_loss_4_8.backward()
            self.disc_optimizer.step()
            self.gen_optimizer.zero_grad()


        _, mpd_gt_feats, mpd_fake_out, mpd_fake_feats = self.model.mpd(target_wav, wav_fake)

        _, msd_gt_feats, msd_fake_out, msd_fake_feats = self.model.msd(target_wav, wav_fake)    
        batch["mpd_fake_out"] = mpd_fake_out
        batch["mpd_fake_feats"] = mpd_fake_feats
        batch["mpd_gt_feats"] = mpd_gt_feats
        batch["msd_fake_out"] = msd_fake_out
        batch["msd_fake_feats"] = msd_fake_feats
        batch["msd_gt_feats"] = msd_gt_feats

        if initial_sr==8000 and target_sr == 16000:
            target_melspec = self.create_mel_spec_8_16(target_wav.squeeze(1))
            batch['mel_spec_hr'] = target_melspec
            mpd_gen_loss_8_16, msd_gen_loss_8_16,\
                mpd_feats_gen_loss_8_16, msd_feats_gen_loss_8_16,\
                mel_spec_loss_8_16, gen_loss_8_16 =\
                    self.criterion.generator_loss_8_16(batch)

        elif initial_sr == 4000 and target_sr == 8000:
            target_melspec = self.create_mel_spec_4_8(target_wav.squeeze(1))
            batch['mel_spec_hr'] = target_melspec

            mpd_gen_loss_4_8, msd_gen_loss_4_8,\
                mpd_feats_gen_loss_4_8, msd_feats_gen_loss_4_8,\
                mel_spec_loss_4_8, gen_loss_4_8 =\
                    self.criterion.generator_loss_4_8(batch)

        if self.is_train:
            self._clip_grad_norm(self.model.generator)
            if initial_sr==8000 and target_sr == 16000:
                gen_loss_8_16.backward()
            elif initial_sr == 4000 and target_sr == 8000:
                gen_loss_4_8.backward()
            self.gen_optimizer.step()

        if initial_sr == 8000 and target_sr==16000:
            batch["disc_loss_8_16"] = disc_loss_8_16
            batch["mpd_gen_loss_8_16"] = mpd_gen_loss_8_16
            batch["msd_gen_loss_8_16"] = msd_gen_loss_8_16
            batch["mel_spec_loss_8_16"] = mel_spec_loss_8_16
            batch["mpd_disc_loss_8_16"] = mpd_disc_loss_8_16
            batch["msd_disc_loss_8_16"] = msd_disc_loss_8_16
            batch["mpd_feats_gen_loss_8_16"] = mpd_feats_gen_loss_8_16
            batch["msd_feats_gen_loss_8_16"] = msd_feats_gen_loss_8_16
            batch["gen_loss_8_16"] = gen_loss_8_16
                
        elif initial_sr==4000 and target_sr==8000:
            batch["disc_loss_4_8"] = disc_loss_4_8
            batch["mpd_gen_loss_4_8"] = mpd_gen_loss_4_8
            batch["msd_gen_loss_4_8"] = msd_gen_loss_4_8
            batch["mel_spec_loss_4_8"] = mel_spec_loss_4_8
            batch["mpd_disc_loss_4_8"] = mpd_disc_loss_4_8
            batch["msd_disc_loss_4_8"] = msd_disc_loss_4_8
            batch["mpd_feats_gen_loss_4_8"] = mpd_feats_gen_loss_4_8
            batch["msd_feats_gen_loss_4_8"] = msd_feats_gen_loss_4_8
            batch["gen_loss_4_8"] = gen_loss_4_8

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch.keys():
                metrics.update(loss_name, batch[loss_name].item())
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(partition='train', idx=0, **batch)
            self.log_audio(partition='train', idx=0, **batch)

        else:
            # Log Stuff
            self.log_spectrogram(partition='val', idx=batch_idx, **batch)
            self.log_audio(partition='val', idx=batch_idx,**batch)


    def log_audio(self, wav_lr, wav_hr,  generated_wav, partition, idx, **batch):
        init_len_lr = batch['initial_len_lr'][0]
        init_len_hr = batch['initial_len_hr'][0]
        initial_sr = batch['initial_sr']
        target_sr = batch['target_sr']
        if partition != 'val':
            self.writer.add_audio(f"initial_wav_lr_{initial_sr}_{target_sr}", wav_lr[0][:, :init_len_lr], initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{initial_sr}_{target_sr}", wav_hr[0][:, :init_len_hr], target_sr)
            self.writer.add_audio(f"generated_wav_{initial_sr}_{target_sr}", generated_wav[0][:, :init_len_hr], target_sr)
        else:
            self.writer.add_audio(f"initial_wav_lr_{initial_sr}_{target_sr}_{idx}", wav_lr[0][:, :init_len_lr], initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{initial_sr}_{target_sr}_{idx}", wav_hr[0][:, :init_len_hr], target_sr)
            self.writer.add_audio(f"generated_wav_{initial_sr}_{target_sr}_{idx}", generated_wav[0][:, :init_len_hr], target_sr)


    def log_spectrogram(self, melspec_lr, melspec_hr,  mel_spec_fake, partition, idx, **batch):
        initial_sr = batch['initial_sr']
        target_sr = batch['target_sr']
        spectrogram_for_plot_real_lr = melspec_lr[0].detach().cpu()[:, :batch['initial_len_melspec_lr'][0]]
        spectrogram_for_plot_real_hr = melspec_hr[0].detach().cpu()[:, :batch['initial_len_melspec_hr'][0]]
        spectrogram_for_plot_fake = mel_spec_fake[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot_real_lr)
        self.writer.add_image(f"melspectrogram_real_lr_{initial_sr}_{target_sr}", image)
        image_hr = plot_spectrogram(spectrogram_for_plot_real_hr)
        self.writer.add_image(f"melspectrogram_real_hr_{initial_sr}_{target_sr}", image_hr)
        image_fake = plot_spectrogram(spectrogram_for_plot_fake)
        self.writer.add_image(f"melspectrogram_fake_{initial_sr}_{target_sr}", image_fake)