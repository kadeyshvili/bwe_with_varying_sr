import torch.nn as nn
from src.model.generator import A2AHiFiPlusGeneratorWithMRF
from src.model.discriminator_p import MultiPeriodDiscriminator
from src.model.discriminator_s import MultiScaleDiscriminator


class HiFiGANWithMRF(nn.Module):
    def __init__(self,
                 generator_config,
                 mpd_config,
                 msd_config):
        super().__init__()
        self.generator = A2AHiFiPlusGeneratorWithMRF(**generator_config)
        self.mpd = MultiPeriodDiscriminator(**mpd_config)
        self.msd = MultiScaleDiscriminator(**msd_config)


    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info