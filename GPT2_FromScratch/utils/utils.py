import numpy as np
import torch


class ResumableRandomSampler(torch.utils.data.Sampler):
    """
    A fixed random permutation of range(n) (same for every run with this seed), rotated to begin at index start.

    The already-seen indices perm[:start] are moved to the end rather than dropped, so len() stays n. On a
    mid-epoch resume Lightning restores how many batches of the epoch were done and ends the epoch when that
    count reaches len(dataloader), so a shorter sampler would end the epoch early and replay its data.
    """

    def __init__(self, n, seed, start=0):
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator).numpy()
        self.indices = np.concatenate([perm[start:], perm[:start]])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
