import torch
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.model.melspec import MelSpectrogram
from src.utils.sr_utils import  get_regime_key




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

        wav_fake = self.model.generator(initial_wav, **batch)

        complex_spec_fake = self.model.generator.get_stft(wav_fake, sampling_rate=target_sr)
        real_fake = complex_spec_fake.real
        imag_fake = complex_spec_fake.imag

        log_amplitude_fake = torch.log(complex_spec_fake.abs() + 1e-5)
        phase_fake = complex_spec_fake.angle()

        if target_wav.shape != wav_fake.shape:
            wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
        batch["generated_wav"] = wav_fake
        batch["real_fake"] = real_fake
        batch["imag_fake"] = imag_fake
        batch["log_amplitude_fake"] = log_amplitude_fake
        batch["phase_fake"] = phase_fake
        
        mel_spec_creator = MelSpectrogram(sr=target_sr).to(self.device)
        mel_spec_fake = mel_spec_creator(wav_fake).squeeze(1)
        batch['mel_spec_fake'] = mel_spec_fake
        if self.is_train:
            self.disc_optimizer.zero_grad()

        mpd_gt_out, _, mpd_fake_out, _ = self.model.mpd(target_wav, wav_fake.detach())

        msd_gt_out, _,  msd_fake_out, _ = self.model.msd(target_wav, wav_fake.detach())
        batch['mpd_gt_out'] = mpd_gt_out
        batch['mpd_fake_out'] = mpd_fake_out
        batch['msd_gt_out'] = msd_gt_out
        batch['msd_fake_out'] = msd_fake_out
        
        #4-8-16, 8-24-48, 4-8, 8-16, 8-16-48,
        if (initial_sr==4000 and target_sr==8000) or (initial_sr==8000 and target_sr==24000):
            mpd_disc_loss, msd_disc_loss, disc_loss = self.criterion.discriminator_loss_2(batch)
        elif (initial_sr==4000 and target_sr==16000) or (initial_sr==8000 and target_sr==48000):
            mpd_disc_loss, msd_disc_loss, disc_loss = self.criterion.discriminator_loss(batch)
        else:
            mpd_disc_loss, msd_disc_loss, disc_loss = self.criterion.discriminator_loss_3(batch)
        



        if self.is_train:
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)

        if self.is_train:
            disc_loss.backward()
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

        target_mel_spec_creator = MelSpectrogram(sr=target_sr).to(self.device)
        target_melspec = target_mel_spec_creator(target_wav.squeeze(1))
        batch['mel_spec_hr'] = target_melspec


        complex_spec = self.model.generator.get_stft(target_wav, sampling_rate=target_sr)
        real_gt = complex_spec.real
        imag_gt = complex_spec.imag
        log_amplitude_gt = torch.log(complex_spec.abs() + 1e-5)
        phase_gt = complex_spec.angle()
        batch['real_gt'] = real_gt
        batch['imag_gt'] = imag_gt
        batch['log_amplitude_gt'] = log_amplitude_gt
        batch['phase_gt'] = phase_gt
        batch['frames'] = phase_gt.shape[-1]
        
        if (initial_sr==4000 and target_sr==8000) or (initial_sr==8000 and target_sr==24000):
            mpd_gen_loss, msd_gen_loss,\
                mpd_feats_gen_loss, msd_feats_gen_loss,\
                mel_spec_loss, loss_stft, phase_loss, amplitude_loss, sisdr_loss, gen_loss =\
                    self.criterion.generator_loss_2(batch)
        elif (initial_sr==4000 and target_sr==16000) or (initial_sr==8000 and target_sr==48000):
            mpd_gen_loss, msd_gen_loss,\
                mpd_feats_gen_loss, msd_feats_gen_loss,\
                mel_spec_loss, loss_stft, phase_loss, amplitude_loss, sisdr_loss, gen_loss =\
                    self.criterion.generator_loss(batch)
        else:
            mpd_gen_loss, msd_gen_loss,\
                mpd_feats_gen_loss, msd_feats_gen_loss,\
                mel_spec_loss, loss_stft, phase_loss, amplitude_loss, sisdr_loss, gen_loss =\
                    self.criterion.generator_loss_3(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.is_train:
            self._clip_grad_norm(self.model.generator)
            gen_loss.backward()
            self.gen_optimizer.step()

        regime_key = get_regime_key(initial_sr, target_sr)
        batch[f"disc_loss_{regime_key}"] = disc_loss
        batch[f"mpd_gen_loss_{regime_key}"] = mpd_gen_loss
        batch[f"msd_gen_loss_{regime_key}"] = msd_gen_loss
        batch[f"mel_spec_loss_{regime_key}"] = mel_spec_loss
        batch[f"mpd_disc_loss_{regime_key}"] = mpd_disc_loss
        batch[f"msd_disc_loss_{regime_key}"] = msd_disc_loss
        batch[f"mpd_feats_gen_loss_{regime_key}"] = mpd_feats_gen_loss
        batch[f"msd_feats_gen_loss_{regime_key}"] = msd_feats_gen_loss
        batch[f"gen_loss_{regime_key}"] = gen_loss
        batch[f"phase_loss_{regime_key}"] = phase_loss
        batch[f"amplitude_loss_{regime_key}"] = amplitude_loss
        batch[f"loss_stft_{regime_key}"] = loss_stft
        batch[f"sisdr_loss_{regime_key}"] = sisdr_loss
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

        # Use fixed samples so the same inputs are logged every epoch
        if mode == "train":
            for regime_key, samples in self.fixed_samples_for_logging.items():
                if regime_key.startswith("val_fixed_"):
                    continue
                for idx, sample in enumerate(samples[:5]):
                    self.log_spectrogram(partition='train', idx=idx, **sample)
                    self.log_audio(partition='train', idx=idx, **sample)
        else:
            val_prefix = f"{mode}_fixed_"
            for fixed_key, samples in self.fixed_samples_for_logging.items():
                if not fixed_key.startswith(val_prefix):
                    continue
                for idx, sample in enumerate(samples[:5]):
                    self.log_spectrogram(partition=mode, idx=idx, **sample)
                    self.log_audio(partition=mode, idx=idx, **sample)


    def log_audio(self, wav_lr, wav_hr,  generated_wav, partition, idx, **batch):
        initial_len_lr = batch['initial_len_lr']
        initial_len_hr = batch['initial_len_hr']
        initial_sr = batch['initial_sr']
        target_sr = batch['target_sr']

        # Log only the first element of the batch to keep a fixed set of samples per epoch
        init_len_lr = initial_len_lr[0]
        init_len_hr = initial_len_hr[0]
        self.writer.add_audio(f"initial_wav_lr_{initial_sr}_{target_sr}_sample{idx}", wav_lr[0][:, :init_len_lr], initial_sr)
        self.writer.add_audio(f"initial_wav_hr_{initial_sr}_{target_sr}_sample{idx}", wav_hr[0][:, :init_len_hr], target_sr)
        self.writer.add_audio(f"generated_wav_{initial_sr}_{target_sr}_sample{idx}", generated_wav[0][:, :init_len_hr], target_sr)


    def log_spectrogram(self, melspec_lr, melspec_hr,  mel_spec_fake, partition, idx, **batch):
        initial_sr = batch['initial_sr']
        target_sr = batch['target_sr']

        initial_len_melspec_lr = batch['initial_len_melspec_lr']
        initial_len_melspec_hr = batch['initial_len_melspec_hr']

        # Log only the first element of the batch to keep a fixed set of samples per epoch
        len_melspec_lr = initial_len_melspec_lr[0]
        len_melspec_hr = initial_len_melspec_hr[0]

        spectrogram_for_plot_real_lr = melspec_lr[0].detach().cpu()[:, :len_melspec_lr]
        spectrogram_for_plot_real_hr = melspec_hr[0].detach().cpu()[:, :len_melspec_hr]
        spectrogram_for_plot_fake = mel_spec_fake[0].detach().cpu()

        image = plot_spectrogram(spectrogram_for_plot_real_lr)
        self.writer.add_image(f"melspectrogram_real_lr_{initial_sr}_{target_sr}_sample{idx}", image)
        image_hr = plot_spectrogram(spectrogram_for_plot_real_hr)
        self.writer.add_image(f"melspectrogram_real_hr_{initial_sr}_{target_sr}_sample{idx}", image_hr)
        image_fake = plot_spectrogram(spectrogram_for_plot_fake)
        self.writer.add_image(f"melspectrogram_fake_{initial_sr}_{target_sr}_sample{idx}", image_fake)