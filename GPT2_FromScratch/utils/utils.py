import torch


class ResumableRandomSampler(torch.utils.data.Sampler):
    """A fixed random permutation of range(n) (same for every run with this seed), starting at index start."""

    def __init__(self, n, seed, start=0):
        generator = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(n, generator=generator)[start:].numpy()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
