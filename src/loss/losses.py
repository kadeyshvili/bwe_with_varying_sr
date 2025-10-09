import torch 
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_gt_output, disc_predicted_output):
        loss = 0
        for gt_output, pred_output in zip(disc_gt_output, disc_predicted_output):
            gt_loss = torch.mean((1 - gt_output) ** 2)
            pred_loss = torch.mean(pred_output ** 2)
            loss += gt_loss + pred_loss
        return loss

        
class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dsc_output):
        loss = 0.0
        for predicted in dsc_output:
            pred_loss = torch.mean((1 - predicted) ** 2)
            loss += pred_loss
        return loss
    

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial, predicted):
        loss = 0
        for disc_initial_feat, disc_pred_feat in zip(initial, predicted):
            for initial_feat, predicted_feat in zip(disc_initial_feat, disc_pred_feat):
                loss += torch.mean(torch.abs(initial_feat - predicted_feat))
        return loss     


class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial_spec, pred_spec):
        return F.l1_loss(pred_spec, initial_spec)
    
class SpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial_spec, pred_spec):
        return F.l1_loss(pred_spec, initial_spec)
    
class HiFiGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.melspec_loss = MelSpectrogramLoss()
        self.fm_loss = FeatureMatchingLoss()
        
        
    def discriminator_loss(self, batch):
        mpd_disc_loss = self.disc_loss(batch["mpd_gt_out"], batch["mpd_fake_out"])
        msd_disc_loss = self.disc_loss(batch["msd_gt_out"], batch["msd_fake_out"])
        return mpd_disc_loss, msd_disc_loss, mpd_disc_loss + msd_disc_loss
        
    def generator_loss(self, batch):
        
        mpd_gen_loss = self.gen_loss(batch["mpd_fake_out"])
        msd_gen_loss = self.gen_loss(batch["msd_fake_out"])   

        #TODO computation of mel specs here with given melSpecComputer as an argument
        #for better generalization to other spectral losses
        mel_spec_loss = self.melspec_loss(batch["mel_spec_hr"], batch["mel_spec_fake"])
        
        mpd_feats_gen_loss = self.fm_loss(batch["mpd_gt_feats"], batch["mpd_fake_feats"])
        msd_feats_gen_loss = self.fm_loss(batch["msd_gt_feats"], batch["msd_fake_feats"])
        
        return mpd_gen_loss, msd_gen_loss, mpd_feats_gen_loss,\
                msd_feats_gen_loss, mel_spec_loss,\
                mpd_gen_loss + msd_gen_loss + 45*mel_spec_loss + 2*mpd_feats_gen_loss + 2*msd_feats_gen_loss
        
        
