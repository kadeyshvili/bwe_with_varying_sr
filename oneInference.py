import warnings

import hydra
import torch
import torchaudio
import librosa
from hydra.utils import instantiate
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="oneinference_config")
def main(config):
    logger = logging.getLogger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Model init.....")
    model = instantiate(config.model).to(device)
    logger.info(str(model))
    logger.info("DONE!")
    
    chkpt_path = config.inferencer.from_pretrained
    logger.info("Loading the weights.....")
    checkpoint = torch.load(chkpt_path, device, weights_only=False)

    if checkpoint["config"]["model"] != config["model"]:
        logger.warning(
            "Warning: Architecture configuration given in the config file is different from that "
            "of the checkpoint. This may yield an exception when state_dict is loaded."
        )

    logger.info("Loading the model.....")
    model.load_state_dict(checkpoint["state_dict"])        

    logger.info("DONE!")
        
    save_path = Path(config.inferencer.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    logger.info("Reading.....")
    audio_tensor, sr = torchaudio.load(config.inferencer.input_path)

    audio_tensor = audio_tensor[0:1, :3*sr]
    logger.info("DONE!")

    logger.info("Resampling.....")
    initial_sr = 4000
    target_sr = 16000 #!!!! HARDCODE
    resStart = 48000

    inp = torch.from_numpy(librosa.resample(audio_tensor.cpu().numpy(), orig_sr=resStart, target_sr=initial_sr, res_type="polyphase"))
    
    torchaudio.save(str(save_path / config.inferencer.original_path), inp.detach().cpu(), sample_rate=initial_sr)

    inp16 = torch.from_numpy(librosa.resample(audio_tensor.cpu().numpy(), orig_sr=resStart, target_sr=target_sr, res_type="polyphase"))

    torchaudio.save(str(save_path / config.inferencer.dowsampled_path),inp16.detach().cpu(),sample_rate=target_sr) 
     
    inp = inp.to(torch.device(device))
    logger.info(inp.shape)
    logger.info("DONE!")
    
    logger.info("Model call.....")
    inp = torch.concatenate([inp[0,:],inp[0,:]],dim=0).unsqueeze(0).unsqueeze(0)
    res = model.generator(inp, initial_sr, target_sr, mode="eval")


    logger.info("DONE!")
    logger.info("Saving...")
    logger.info(res.squeeze([0]).shape)
    torchaudio.save(str(save_path / config.inferencer.generated_path), res.squeeze([0]).detach().cpu(), sample_rate=target_sr)
    logger.info("DONE!")
    
    
    

if __name__ == "__main__":
    main()
