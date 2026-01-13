"""
Utility functions for sample rate handling and regime determination.
"""
import torch


def get_sr_ratio(initial_sr: int, target_sr: int) -> int:
    """
    Calculate the ratio between target and initial sample rates.
    
    Args:
        initial_sr: Initial sample rate
        target_sr: Target sample rate
    
    Returns:
        Ratio (e.g., 2 for 4->8, 4 for 4->16)
    """
    return target_sr // initial_sr


def get_regime_key(initial_sr: int, target_sr: int) -> str:
    """
    Get regime key string based on initial and target sample rates.
    
    Args:
        initial_sr: Initial sample rate
        target_sr: Target sample rate
    
    Returns:
        Regime key string (e.g., "4_8", "8_16", "8_24", "4_16")
    """
    return f"{initial_sr // 1000}_{target_sr // 1000}"


def get_intermediate_sr(initial_sr: int, target_sr: int) -> int:
    """
    Get intermediate sample rate for multi-step upsampling.
    For ratio 2 or 3: returns target_sr (no intermediate needed)
    For ratio >= 4: returns intermediate SR (initial_sr * 2) - one additional block
    
    Examples:
        - 4->8 (ratio=2): returns 8 (no intermediate)
        - 8->16 (ratio=2): returns 16 (no intermediate)
        - 8->24 (ratio=3): returns 24 (no intermediate, direct 8->24)
        - 4->16 (ratio=4): returns 8 (4->8->16)
        - 4->24 (ratio=6): returns 8 (4->8->24)
    
    Args:
        initial_sr: Initial sample rate
        target_sr: Target sample rate
    
    Returns:
        Intermediate sample rate
    """
    ratio = get_sr_ratio(initial_sr, target_sr)
    if ratio == 2 or ratio == 3:
        return target_sr  # No intermediate needed for direct upsampling
    elif ratio >= 4:
        # For ratio >= 4, use one intermediate step: initial_sr -> initial_sr*2 -> target_sr
        return initial_sr * 2
    else:
        raise ValueError(f"Unsupported ratio: {ratio}. Ratio must be >= 2.")


def get_num_blocks(initial_sr: int, target_sr: int) -> int:
    """
    Get number of upsampling blocks needed based on sample rate ratio.
    
    Examples:
        - 4->8 (ratio=2): 1 block
        - 8->16 (ratio=2): 1 block
        - 8->24 (ratio=3): 1 block (direct 8->24)
        - 4->16 (ratio=4): 2 blocks (4->8->16)
        - 4->24 (ratio=6): 2 blocks (4->8->24)
    
    Args:
        initial_sr: Initial sample rate
        target_sr: Target sample rate
    
    Returns:
        Number of blocks needed (1 for ratio 2 or 3, 2 for ratio >= 4)
    """
    ratio = get_sr_ratio(initial_sr, target_sr)
    if ratio == 2 or ratio == 3:
        return 1  # Direct upsampling: 4->8, 8->16, 8->24
    elif ratio >= 4:
        return 2  # Two blocks with intermediate step
    else:
        raise ValueError(f"Unsupported ratio: {ratio}. Ratio must be >= 2.")


def create_band_mask(initial_sr: int, target_sr: int, batch_size: int, device) -> torch.Tensor:
    """
    Create band mask for frequency filtering.
    
    Args:
        initial_sr: Initial sample rate
        target_sr: Target sample rate
        batch_size: Batch size
        device: Device to create tensor on
    
    Returns:
        Band mask tensor
    """
    highcut = initial_sr // 2
    nyq = 0.5 * target_sr
    hi = highcut / nyq
    fft_size = 1024 // 2 + 1
    
    band_mask = torch.zeros(fft_size, dtype=torch.float)
    band_mask[:int(hi * fft_size)] = 1
    band_mask = band_mask.unsqueeze(0).unsqueeze(0)
    band_mask = band_mask.repeat(batch_size, 2, 1).to(device)
    
    return band_mask
