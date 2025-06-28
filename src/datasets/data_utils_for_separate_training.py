from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate_for_separate import collate_fn
from src.utils.init_utils import set_worker_seed
from src.datasets import SRConsistentBatchSampler
import torch


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader

def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        text_encoder (CTCTextEncoder): instance of the text encoder
            for the datasets.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """


    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        dataset = instantiate(
            config.datasets[dataset_partition]
        )  # instance transforms are defined inside

        assert config.dataloader[dataset_partition].batch_size <= len(dataset), (
            f"The batch size ({config.dataloader[dataset_partition].batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        batch_sampler = SRConsistentBatchSampler(dataset, config.dataloader[dataset_partition].batch_size)
        partition_dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_seed,
            num_workers=config.dataloader[dataset_partition].num_workers,
            pin_memory=config.dataloader[dataset_partition].pin_memory
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders