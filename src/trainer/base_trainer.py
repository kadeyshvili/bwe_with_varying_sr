from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import itertools
from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.model.melspec import  MelSpectrogram
import pandas as pd
import numpy as np
from hydra.utils import instantiate


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        gen_optimizer,
        disc_optimizer,
        gen_lr_scheduler,
        disc_lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            text_encoder (CTCTextEncoder): text encoder.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler
        self.create_mel_spec = MelSpectrogram(sr=config.datasets.train.target_sr).to(self.device)
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self._last_epoch = 0  
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs


        self.save_period = (
            self.cfg_trainer.save_period
        )  
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        ) 

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "generator_grad_norm",
            "discriminator_grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

           
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.generator.train()
        self.model.msd.train()
        self.model.mpd.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )

            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("generator_grad_norm", self._get_grad_norm(self.model.generator.parameters()))
            self.train_metrics.update("discriminator_grad_norm", self._get_grad_norm(itertools.chain(self.model.mpd.parameters(),\
                                                                                                      self.model.msd.parameters())))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator Loss: {:.6f}, Discriminator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["gen_loss"].item(), batch['disc_loss'].item()
                    )
                )
                self.writer.add_scalar(
                    "learning_rate_generator", self.gen_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning_rate_discriminator", self.disc_lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break
        self.gen_lr_scheduler.step()
        self.disc_lr_scheduler.step()

        logs = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.generator.eval()
        self.model.mpd.eval()
        self.model.msd.eval()
        self.evaluation_metrics.reset()
        self.writer.mode = part
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
                if self.config.dataloader['val'].batch_size == 1:
                    self._log_batch(
                        batch_idx, batch, part
                    )
            self.writer.set_step(epoch * self.epoch_len, part)
            if self.config.dataloader['val'].batch_size != 1:
                self._log_batch(
                    batch_idx, batch, part
                )
        for i in range(len(self.metrics['inference'])):
            self.evaluation_metrics.update(self.metrics['inference'][i].name, np.mean(self.metrics['inference'][i].result['mean']))
            self.metrics['inference'][i].result['mean'] = []
            self.metrics['inference'][i].result['std'] = []

        self._log_scalars(self.evaluation_metrics)
        
        

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                # stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch



    def _clip_grad_norm(self, model):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, parameters, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if len(grads) == 0:
            return torch.tensor(0.0, device=self.device)  # safe fallback
        return torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "gen_lr_scheduler": self.gen_lr_scheduler.state_dict(),
            "disc_lr_scheduler": self.disc_lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.
        """

        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        for name, param in self.model.generator.named_parameters():
            if "waveunet" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params_gen = list(
            filter(lambda p: p.requires_grad, self.model.generator.parameters())
        )
        trainable_params_disc = list(
            filter(lambda p: p.requires_grad, self.model.mpd.parameters())
        ) + list(
            filter(lambda p: p.requires_grad, self.model.msd.parameters())
        )

        self.gen_optimizer = instantiate(self.config["gen_optimizer"], params=trainable_params_gen)
        self.disc_optimizer = instantiate(self.config["disc_optimizer"], params=trainable_params_disc)

        self.gen_lr_scheduler = instantiate(self.config["gen_lr_scheduler"], optimizer=self.gen_optimizer)
        self.disc_lr_scheduler = instantiate(self.config["disc_lr_scheduler"], optimizer=self.disc_optimizer)

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

        def count_params(named_params):
            frozen, trainable = [], []
            for n, p in named_params:
                (trainable if p.requires_grad else frozen).append(n)
            return frozen, trainable

        gen_frozen, gen_trainable = count_params(self.model.generator.named_parameters())
        mpd_frozen, mpd_trainable = count_params(self.model.mpd.named_parameters())
        msd_frozen, msd_trainable = count_params(self.model.msd.named_parameters())

        self.logger.info(f"[Generator] frozen={len(gen_frozen)}, trainable={len(gen_trainable)}")
        self.logger.info(f"    Trainable example: {gen_trainable[:10]}")
        self.logger.info(f"[MPD] frozen={len(mpd_frozen)}, trainable={len(mpd_trainable)}")
        self.logger.info(f"[MSD] frozen={len(msd_frozen)}, trainable={len(msd_trainable)}")

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"): 
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device, weights_only=False)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
