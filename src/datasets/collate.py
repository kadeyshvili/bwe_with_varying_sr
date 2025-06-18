import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    all_wavs_lr = []
    all_wavs_hr = []
    all_melspecs_lr = []
    all_melspecs_hr = []
    max_len_wav_lr = 0
    max_len_wav_hr = 0
    max_len_spec_lr = 0
    max_len_spec_hr = 0
    paths_lr = []
    paths_hr = []
    initial_lens_lr = []
    initial_lens_hr = []
    initial_len_melspec_lr = []
    initial_len_melspec_hr = []
    mode = None
    reference_wav = []
    initial_len_reference_wav = []
    max_len_reference_wav = 0
    for item in dataset_items:
        mode = item['mode']
        reference_wav.append(item['reference_wav'])
        initial_len_reference_wav.append(item['reference_wav'].shape[-1])
        max_len_reference_wav = max(max_len_reference_wav, item['reference_wav'].shape[-1])
        paths_lr.append(item['path_lr'])
        paths_hr.append(item['path_hr'])
        all_wavs_lr.append(item['wav_lr'].squeeze(0))
        all_wavs_hr.append(item['wav_hr'].squeeze(0))
        all_melspecs_lr.append(item['melspec_lr'])
        all_melspecs_hr.append(item['melspec_hr'])
        max_len_wav_lr = max(len(item['wav_lr'].squeeze(0)), max_len_wav_lr)
        max_len_wav_hr = max(len(item['wav_hr'].squeeze(0)), max_len_wav_hr)
        max_len_spec_lr =  max(item['melspec_lr'].shape[-1], max_len_spec_lr)
        max_len_spec_hr =  max(item['melspec_hr'].shape[-1], max_len_spec_hr)
        initial_len_melspec_lr.append(item['melspec_lr'].shape[1])
        initial_len_melspec_hr.append(item['melspec_hr'].shape[1])
        initial_lens_lr.append(item['wav_lr'].shape[1])
        initial_lens_hr.append(item['wav_hr'].shape[1])

    result_batch['initial_len_lr'] = initial_lens_lr
    result_batch['initial_len_hr'] = initial_lens_hr
    result_batch['initial_len_melspec_lr'] = initial_len_melspec_lr
    result_batch['initial_len_melspec_hr'] = initial_len_melspec_hr
    padded_wavs_lr = torch.stack([F.pad(wav, (0, max_len_wav_lr - wav.shape[0]), value=0) for wav in all_wavs_lr])    
    padded_wavs_hr = torch.stack([F.pad(wav, (0, max_len_wav_hr - wav.shape[0]), value=0) for wav in all_wavs_hr])
    padded_specs_lr = torch.stack([F.pad(spec, (0, max_len_spec_lr - spec.shape[-1], 0, 0)) for spec in all_melspecs_lr])
    padded_specs_hr = torch.stack([F.pad(spec, (0, max_len_spec_hr - spec.shape[-1], 0, 0)) for spec in all_melspecs_hr])
    result_batch['wav_lr'] = padded_wavs_lr.unsqueeze(1)
    result_batch['wav_hr'] = padded_wavs_hr.unsqueeze(1)
    result_batch['melspec_lr'] = padded_specs_lr
    result_batch['melspec_hr'] = padded_specs_hr
    result_batch['paths_lr'] = paths_lr
    result_batch['paths_hr'] = paths_hr
    result_batch['mode'] = mode
    result_batch['reference_wav'] = torch.stack([F.pad(wav, (0, max_len_reference_wav - wav.shape[0]), value=0) for wav in reference_wav]).unsqueeze(1) 
    return result_batch