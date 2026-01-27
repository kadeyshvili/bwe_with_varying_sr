import torch
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.model import HiFiGANWithMRF
from src.model.melspec import MelSpectrogram
from src.utils.sr_utils import get_sr_ratio, get_regime_key
from hydra.utils import instantiate




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

        if target_wav.shape != wav_fake.shape:
            wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
        batch["generated_wav"] = wav_fake
        
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
        
        if (initial_sr==4000 and target_sr==8000) or (initial_sr==8000 and target_sr==24000):
            mpd_gen_loss, msd_gen_loss,\
                mpd_feats_gen_loss, msd_feats_gen_loss,\
                mel_spec_loss, gen_loss =\
                    self.criterion.generator_loss_2(batch)
        elif (initial_sr==4000 and target_sr==16000) or (initial_sr==8000 and target_sr==48000):
            mpd_gen_loss, msd_gen_loss,\
                mpd_feats_gen_loss, msd_feats_gen_loss,\
                mel_spec_loss, gen_loss =\
                    self.criterion.generator_loss(batch)
        else:
            mpd_gen_loss, msd_gen_loss,\
                mpd_feats_gen_loss, msd_feats_gen_loss,\
                mel_spec_loss, gen_loss =\
                    self.criterion.generator_loss_3(batch)

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
            for regime_key in self.samples_for_logging.keys():
                samples = self.samples_for_logging.get(regime_key, [])
                for idx, sample in enumerate(samples[:5]):
                    self.log_spectrogram(partition='train', idx=idx, **sample)
                    self.log_audio(partition='train', idx=idx, **sample)
            self.samples_for_logging = {key: [] for key in self.samples_for_logging.keys()}
        else:
            for regime_key in self.samples_for_logging.keys():
                samples = self.samples_for_logging.get(regime_key, [])
                for idx, sample in enumerate(samples[:5]):
                    self.log_spectrogram(partition='val', idx=idx, **sample)
                    self.log_audio(partition='val', idx=idx, **sample)
            self.samples_for_logging = {key: [] for key in self.samples_for_logging.keys()}


    def log_audio(self, wav_lr, wav_hr,  generated_wav, partition, idx, **batch):
        initial_len_lr = batch['initial_len_lr']
        initial_len_hr = batch['initial_len_hr']
        for i in range(len(initial_len_lr)):

            init_len_lr = initial_len_lr[i]
            init_len_hr = initial_len_hr[i]
            
            initial_sr = batch['initial_sr']
            target_sr = batch['target_sr']
            regime = batch.get('regime', None)
            
            self.writer.add_audio(f"initial_wav_lr_{initial_sr}_{target_sr}_{i}", wav_lr[i][:, :init_len_lr], initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{initial_sr}_{target_sr}_{i}", wav_hr[i][:, :init_len_hr], target_sr)
            self.writer.add_audio(f"generated_wav_{initial_sr}_{target_sr}_{i}", generated_wav[i][:, :init_len_hr], target_sr)


    def log_spectrogram(self, melspec_lr, melspec_hr,  mel_spec_fake, partition, idx, **batch):
        initial_sr = batch['initial_sr']
        target_sr = batch['target_sr']
        regime = batch.get('regime', None)

        
        initial_len_melspec_lr = batch['initial_len_melspec_lr']
        initial_len_melspec_hr = batch['initial_len_melspec_hr']
        for i in range(len(initial_len_melspec_lr)):

            
            len_melspec_lr =  initial_len_melspec_lr[i]
            len_melspec_hr = initial_len_melspec_hr[i]

            spectrogram_for_plot_real_lr = melspec_lr[i].detach().cpu()[:, :len_melspec_lr]
            spectrogram_for_plot_real_hr = melspec_hr[i].detach().cpu()[:, :len_melspec_hr]
            spectrogram_for_plot_fake = mel_spec_fake[i].detach().cpu()
            
            image = plot_spectrogram(spectrogram_for_plot_real_lr)
            self.writer.add_image(f"melspectrogram_real_lr_{initial_sr}_{target_sr}_{i}", image)
            image_hr = plot_spectrogram(spectrogram_for_plot_real_hr)
            self.writer.add_image(f"melspectrogram_real_hr_{initial_sr}_{target_sr}_{i}", image_hr)
            image_fake = plot_spectrogram(spectrogram_for_plot_fake)
            self.writer.add_image(f"melspectrogram_fake_{initial_sr}_{target_sr}_{i}", image_fake)