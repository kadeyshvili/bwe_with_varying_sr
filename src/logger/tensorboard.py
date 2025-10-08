
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TensorboardFramework:
    
    def __init__(self, 
        logger,
        project_config,
        project_name,
        log_dir="./tblogs/", **kwargs):
        
        self.logger = logger
        self.project_name = project_name
        self.project_config = project_config#????
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.step = 0
        self.timer = datetime.now()


    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the Assets & Artifacts section
        inside the models/checkpoints directory.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """
        # seems not relevant for tb
        pass
    
    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.writer.add_scalar(self._object_name(scalar_name), scalar, self.step)
        
    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | Tensor | ndarray | list[tuple] | Image): image
                in the CometML-friendly format.
        """
        pass #does not work because of image format
        #self.writer.add_image(self._object_name(image_name), np.array(image), global_step=self.step)

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (Path | ndarray): audio in the CometML-friendly format.
            sample_rate (int): audio sample rate.
        """
        audio = audio.detach().cpu().numpy().T
        self.writer.add_audio(
            tag=self._object_name(audio_name),
            snd_tensor=audio,
            global_step=self.step,
            sample_rate=sample_rate,
        )
        
            
    