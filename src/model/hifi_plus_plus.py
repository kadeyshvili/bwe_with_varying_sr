import torch.nn as nn
from src.model.generator import A2AHiFiPlusPlus
from src.model.discriminator_p import MultiPeriodDiscriminator
from src.model.discriminator_s import MultiScaleDiscriminator


class HiFiPlusPlusGAN(nn.Module):
    def __init__(self,
                 generator_config,
                 mpd_config,
                 msd_config):
        super().__init__()
        self.generator = A2AHiFiPlusPlus(**generator_config)
        self.mpd = MultiPeriodDiscriminator(**mpd_config)
        self.msd = MultiScaleDiscriminator(**msd_config)


    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        gen_parameters = sum(
            [p.numel() for p in self.generator.parameters() if p.requires_grad]
        )
        msd_parameters = sum(
            [p.numel() for p in self.msd.parameters() if p.requires_grad]
        )
        mpd_parameters = sum(
            [p.numel() for p in self.mpd.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = result_info + f"\nGen: {gen_parameters}"
        result_info = result_info + f"\nMSD: {msd_parameters}"
        result_info = result_info + f"\nMPD: {mpd_parameters}"
        return result_info