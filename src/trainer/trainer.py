from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.metrics.calculate_metrics import calculate_all_metrics
from src.model.generator import mel_spectrogram



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
        initial_sr = self.config.datasets.train.initial_sr
        target_sr = self.config.datasets.train.target_sr
        wav_fake = self.model.generator(initial_wav, initial_sr, target_sr, **batch)
 
        if target_wav.shape != wav_fake.shape:
            wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
        batch["generated_wav"] = wav_fake
        mel_spec_fake = self.create_mel_spec(wav_fake).squeeze(1)
        batch['mel_spec_fake'] = mel_spec_fake
        if self.is_train:
            self.disc_optimizer.zero_grad()

        mpd_gt_out, _, mpd_fake_out, _ = self.model.mpd(target_wav, wav_fake.detach())

        msd_gt_out, _,  msd_fake_out, _ = self.model.msd(target_wav, wav_fake.detach())

        mpd_disc_loss = self.criterion.discriminator_loss(mpd_gt_out, mpd_fake_out)
        msd_disc_loss = self.criterion.discriminator_loss(msd_gt_out, msd_fake_out)
        disc_loss = mpd_disc_loss + msd_disc_loss


        if self.is_train:
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)

        if self.is_train:
            disc_loss.backward()
            self.disc_optimizer.step()
            self.gen_optimizer.zero_grad()


        _, mpd_gt_feats, mpd_fake_out, mpd_fake_feats = self.model.mpd(target_wav, wav_fake)

        _, msd_gt_features, msd_fake_out, msd_fake_feats = self.model.msd(target_wav, wav_fake)     

        mpd_gen_loss = self.criterion.generator_loss(mpd_fake_out)
        msd_gen_loss = self.criterion.generator_loss(msd_fake_out)

        target_melspec = mel_spectrogram(target_wav.squeeze(1), 1024, 80, 16000, 256, 1024, 0, 8000)

        mel_spec_loss = self.criterion.melspec_loss(target_melspec, mel_spec_fake)
        
        mpd_feats_gen_loss = self.criterion.fm_loss(mpd_gt_feats, mpd_fake_feats)
        msd_feats_gen_loss = self.criterion.fm_loss(msd_gt_features, msd_fake_feats)

        gen_loss = mpd_gen_loss + msd_gen_loss + mel_spec_loss + mpd_feats_gen_loss + msd_feats_gen_loss


        if self.is_train:
            self._clip_grad_norm(self.model.generator)
            gen_loss.backward()
            self.gen_optimizer.step()


        batch["mpd_disc_loss"] = mpd_disc_loss
        batch["msd_disc_loss"] = msd_disc_loss
        batch["disc_loss"] = disc_loss
        batch["mpd_gen_loss"] = mpd_gen_loss
        batch["msd_gen_loss"] = msd_gen_loss
        batch["mel_spec_loss"] = mel_spec_loss
        batch["mpd_feats_gen_loss"] = mpd_feats_gen_loss
        batch["msd_feats_gen_loss"] = msd_feats_gen_loss
        batch["gen_loss"] = gen_loss
    

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        if not self.is_train:
            calculate_all_metrics(batch['generated_wav'], batch['wav_hr'], self.metrics["inference"], self.config.datasets.val.initial_sr, self.config.datasets.val.target_sr)

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
        if mode == "train": 
            self.log_spectrogram(partition='train', idx=0, **batch)
            self.log_audio(partition='train', idx=0, **batch)

        else:
            self.log_spectrogram(partition='val', idx=batch_idx, **batch)
            self.log_audio(partition='val', idx=batch_idx,**batch)


    def log_audio(self, wav_lr, wav_hr,  generated_wav, partition, idx, **batch):
        init_len_lr = batch['initial_len_lr'][0]
        init_len_hr = batch['initial_len_hr'][0]
        if partition != 'val':
            self.writer.add_audio("initial_wav_lr", wav_lr[0][:, :init_len_lr], self.config.datasets.train.initial_sr)
            self.writer.add_audio("initial_wav_hr", wav_hr[0][:, :init_len_hr], self.config.datasets.train.target_sr)
            self.writer.add_audio("generated_wav", generated_wav[0][:, :init_len_hr], self.config.datasets.train.target_sr)
        else:
            self.writer.add_audio(f"initial_wav_lr_{idx}", wav_lr[0][:, :init_len_lr], self.config.datasets.val.initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{idx}", wav_hr[0][:, :init_len_hr], self.config.datasets.val.target_sr)
            self.writer.add_audio(f"generated_wav_{idx}", generated_wav[0][:, :init_len_hr], self.config.datasets.val.target_sr)


    def log_spectrogram(self, melspec_lr, melspec_hr,  mel_spec_fake, partition, idx, **batch):
        spectrogram_for_plot_real_lr = melspec_lr[0].detach().cpu()[:, :batch['initial_len_melspec_lr'][0]]
        spectrogram_for_plot_real_hr = melspec_hr[0].detach().cpu()[:, :batch['initial_len_melspec_hr'][0]]
        spectrogram_for_plot_fake = mel_spec_fake[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot_real_lr)
        self.writer.add_image("melspectrogram_real_lr", image)
        image_hr = plot_spectrogram(spectrogram_for_plot_real_hr)
        self.writer.add_image("melspectrogram_real_hr", image_hr)
        image_fake = plot_spectrogram(spectrogram_for_plot_fake)
        self.writer.add_image("melspectrogram_fake", image_fake)

