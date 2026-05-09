"""Dataset and dataloaders for (a + b) mod p."""
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ModularAdditionDataset(Dataset):
    def __init__(self, p: int):
        self.p = p
        pairs = [(a, b) for a in range(p) for b in range(p)]
        self.a = torch.tensor([x[0] for x in pairs], dtype=torch.long)
        self.b = torch.tensor([x[1] for x in pairs], dtype=torch.long)
        self.c = (self.a + self.b) % p

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.c[idx]


def make_dataloaders(p: int = 97, train_frac: float = 0.30,
                     batch_size: int = 128, seed: int = 0):
    """Return (train_loader, test_loader)."""
    dataset = ModularAdditionDataset(p)
    n_train = int(len(dataset) * train_frac)
    n_test = len(dataset) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)
    kw = dict(batch_size=batch_size, drop_last=False, pin_memory=False)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(test_ds,  shuffle=False, **kw))
