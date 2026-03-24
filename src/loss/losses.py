import torch 
import torch.nn as nn
import numpy as np
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
    

class STFT_consistency_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_fake, imag_fake, real_gt, imag_gt):
        loss = torch.mean(
            torch.mean((real_gt - real_fake) ** 2 + (imag_gt - imag_fake) ** 2, (1, 2))
        )
        return loss


class Amplitude_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_amplitude_gt, log_amplitude_fake):
        MSELoss = torch.nn.MSELoss()
        amplitude_loss = MSELoss(log_amplitude_gt, log_amplitude_fake)
        return amplitude_loss
        


class Phase_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def anti_wrapping_function(self, x):
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    def forward(self, phase_gt, phase_fake, n_fft, frames):
        GD_matrix = (
        torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
        - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
        - torch.eye(n_fft // 2 + 1)
        )
        GD_matrix = GD_matrix.to(phase_fake.device)

        GD_r = torch.matmul(phase_gt.permute(0, 2, 1), GD_matrix)
        GD_g = torch.matmul(phase_fake.permute(0, 2, 1), GD_matrix)

        PTD_matrix = (
            torch.triu(torch.ones(frames, frames), diagonal=1)
            - torch.triu(torch.ones(frames, frames), diagonal=2)
            - torch.eye(frames)
        )
        PTD_matrix = PTD_matrix.to(phase_fake.device)

        PTD_r = torch.matmul(phase_gt, PTD_matrix)
        PTD_g = torch.matmul(phase_fake, PTD_matrix)

        IP_loss = torch.mean(self.anti_wrapping_function(phase_gt - phase_fake))
        GD_loss = torch.mean(self.anti_wrapping_function(GD_r - GD_g))
        PTD_loss = torch.mean(self.anti_wrapping_function(PTD_r - PTD_g))

        return IP_loss + GD_loss +PTD_loss



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
        # Loss modules for ratio the first part
        self.disc_loss_ratio_2 = DiscriminatorLoss()
        self.gen_loss_ratio_2 = GeneratorLoss()
        self.melspec_loss_ratio_2 = MelSpectrogramLoss()
        self.fm_loss_ratio_2 = FeatureMatchingLoss()
        self.stft_consistency_loss_ratio_2 = STFT_consistency_loss()
        self.amplitude_loss_ratio_2 = Amplitude_loss()
        self.phase_loss_ratio_2 = Phase_loss()
        

        # Loss modules for ratio the second part
        self.disc_loss_ratio_3 = DiscriminatorLoss()
        self.gen_loss_ratio_3 = GeneratorLoss()
        self.melspec_loss_ratio_3= MelSpectrogramLoss()
        self.fm_loss_ratio_3 = FeatureMatchingLoss()
        self.stft_consistency_loss_ratio_3 = STFT_consistency_loss()
        self.amplitude_loss_ratio_3 = Amplitude_loss()
        self.phase_loss_ratio_3 = Phase_loss()
        # Loss modules for full part
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.melspec_loss = MelSpectrogramLoss()
        self.fm_loss = FeatureMatchingLoss()
        self.stft_consistency_loss_ratio = STFT_consistency_loss()
        self.amplitude_loss_ratio = Amplitude_loss()
        self.phase_loss_ratio = Phase_loss()

        
    def discriminator_loss_2(self, batch):
        mpd_disc_loss = self.disc_loss_ratio_2(batch["mpd_gt_out"], batch["mpd_fake_out"])
        msd_disc_loss = self.disc_loss_ratio_2(batch["msd_gt_out"], batch["msd_fake_out"])
        return mpd_disc_loss, msd_disc_loss, mpd_disc_loss + msd_disc_loss
    
    def discriminator_loss_3(self, batch):
        mpd_disc_loss = self.disc_loss_ratio_3(batch["mpd_gt_out"], batch["mpd_fake_out"])
        msd_disc_loss = self.disc_loss_ratio_3(batch["msd_gt_out"], batch["msd_fake_out"])
        return mpd_disc_loss, msd_disc_loss, mpd_disc_loss + msd_disc_loss
    
    def discriminator_loss(self, batch):
        mpd_disc_loss = self.disc_loss(batch["mpd_gt_out"], batch["mpd_fake_out"])
        msd_disc_loss = self.disc_loss(batch["msd_gt_out"], batch["msd_fake_out"])
        return mpd_disc_loss, msd_disc_loss, mpd_disc_loss + msd_disc_loss
    
    def generator_loss_2(self, batch):
        mpd_gen_loss = self.gen_loss_ratio_2(batch["mpd_fake_out"])
        msd_gen_loss = self.gen_loss_ratio_2(batch["msd_fake_out"])   

        mel_spec_loss = self.melspec_loss_ratio_2(batch["mel_spec_hr"], batch["mel_spec_fake"])
        
        mpd_feats_gen_loss = self.fm_loss_ratio_2(batch["mpd_gt_feats"], batch["mpd_fake_feats"])
        msd_feats_gen_loss = self.fm_loss_ratio_2(batch["msd_gt_feats"], batch["msd_fake_feats"])


        loss_real_part = F.l1_loss(batch['real_gt'], batch['real_fake'])
        loss_imag_part = F.l1_loss(batch['imag_gt'], batch['imag_fake'])
        stft_consistency_loss = self.stft_consistency_loss_ratio_2(batch["real_fake"], batch["imag_fake"], batch["real_gt"], batch["imag_gt"])
        loss_stft = stft_consistency_loss + 2.25 * (loss_real_part + loss_imag_part)
        phase_loss = self.phase_loss_ratio_2(batch["phase_gt"], batch["phase_fake"], 1024, batch["frames"])
        amplitude_loss = self.amplitude_loss_ratio_2(batch["log_amplitude_gt"], batch["log_amplitude_fake"])
        
        return mpd_gen_loss, msd_gen_loss, mpd_feats_gen_loss,\
                msd_feats_gen_loss, mel_spec_loss, loss_stft,phase_loss,amplitude_loss,\
                20*loss_stft + mpd_gen_loss + msd_gen_loss + 45*mel_spec_loss + 2*mpd_feats_gen_loss + 2*msd_feats_gen_loss + 100*phase_loss + 45*amplitude_loss
    

    def generator_loss_3(self, batch):
        mpd_gen_loss = self.gen_loss_ratio_3(batch["mpd_fake_out"])
        msd_gen_loss = self.gen_loss_ratio_3(batch["msd_fake_out"])   

        mel_spec_loss = self.melspec_loss_ratio_3(batch["mel_spec_hr"], batch["mel_spec_fake"])
        
        mpd_feats_gen_loss = self.fm_loss_ratio_3(batch["mpd_gt_feats"], batch["mpd_fake_feats"])
        msd_feats_gen_loss = self.fm_loss_ratio_3(batch["msd_gt_feats"], batch["msd_fake_feats"])

        loss_real_part = F.l1_loss(batch['real_gt'], batch['real_fake'])
        loss_imag_part = F.l1_loss(batch['imag_gt'], batch['imag_fake'])
        stft_consistency_loss = self.stft_consistency_loss_ratio(batch["real_fake"], batch["imag_fake"], batch["real_gt"], batch["imag_gt"])
        loss_stft = stft_consistency_loss + 2.25 * (loss_real_part + loss_imag_part)
        phase_loss = self.phase_loss_ratio_3(batch["phase_gt"], batch["phase_fake"], 1024, batch["frames"])
        amplitude_loss = self.amplitude_loss_ratio_3(batch["log_amplitude_gt"], batch["log_amplitude_fake"])
        
        
        return mpd_gen_loss, msd_gen_loss, mpd_feats_gen_loss,\
                msd_feats_gen_loss, mel_spec_loss, loss_stft,phase_loss,amplitude_loss,\
                20*loss_stft + mpd_gen_loss + msd_gen_loss + 45*mel_spec_loss + 2*mpd_feats_gen_loss + 2*msd_feats_gen_loss + 100*phase_loss + 45*amplitude_loss

    def generator_loss(self, batch):
        mpd_gen_loss = self.gen_loss(batch["mpd_fake_out"])
        msd_gen_loss = self.gen_loss(batch["msd_fake_out"])   

        mel_spec_loss = self.melspec_loss(batch["mel_spec_hr"], batch["mel_spec_fake"])
        
        mpd_feats_gen_loss = self.fm_loss(batch["mpd_gt_feats"], batch["mpd_fake_feats"])
        msd_feats_gen_loss = self.fm_loss(batch["msd_gt_feats"], batch["msd_fake_feats"])

        loss_real_part = F.l1_loss(batch['real_gt'], batch['real_fake'])
        loss_imag_part = F.l1_loss(batch['imag_gt'], batch['imag_fake'])
        stft_consistency_loss = self.stft_consistency_loss_ratio(batch["real_fake"], batch["imag_fake"], batch["real_gt"], batch["imag_gt"])
        loss_stft = stft_consistency_loss + 2.25 * (loss_real_part + loss_imag_part)
        phase_loss = self.phase_loss_ratio(batch["phase_gt"], batch["phase_fake"], 1024, batch["frames"])
        amplitude_loss = self.amplitude_loss_ratio(batch["log_amplitude_gt"], batch["log_amplitude_fake"])
        

        return mpd_gen_loss, msd_gen_loss, mpd_feats_gen_loss,\
                msd_feats_gen_loss, mel_spec_loss, loss_stft,phase_loss,amplitude_loss,\
                20*loss_stft + mpd_gen_loss + msd_gen_loss + 45*mel_spec_loss + 2*mpd_feats_gen_loss + 2*msd_feats_gen_loss + 100*phase_loss + 45*amplitude_loss
        
        
        
