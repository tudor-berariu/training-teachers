from math import ceil
from typing import Iterator, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# TODO: Other datasets than FashionMNIST


ORIGINAL_SIZE = {"FashionMNIST": (1, 28, 28)}
MEAN_STD = {"FashionMNIST": {(1, 32, 32): (0.2190, 0.3318,),
                             (1, 28, 28): (0.2860, 0.3530,)}}


Size = Tuple[int, int, int]
Padding = Tuple[int, int, int, int]


class InMemoryDataLoader(object):
    # Only for small datasets

    def __init__(self, loader: DataLoader,
                 batch_size: int,
                 shuffle: bool = False) -> None:
        full_data, full_target = None, None
        self.batch_size = batch_size
        self.shuffle = shuffle
        for data, target in loader:
            if full_data is not None:
                full_data = torch.cat((full_data, data), dim=0)
                full_target = torch.cat((full_target, target), dim=0)
            else:
                full_data, full_target = data, target
        self.data = full_data
        self.target = full_target
        self.length = len(self.data)
        self.idxs, self.offset = None, None

    def to(self, device: torch.device) -> None:
        self.data, self.target = self.data.to(device), self.target.to(device)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        if self.shuffle:
            self.idxs = torch.randperm(self.data.size(0))
        self.offset = 0
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        start = self.offset
        end = min(self.offset + self.batch_size, self.length)
        if start >= end:
            raise StopIteration
        self.offset = end

        if self.shuffle:
            idxs = self.idxs[start:end]
            return self.data[idxs], self.target[idxs]

        return self.data[start:end], self.target[start:end]

    def __len__(self) -> int:
        return int(ceil(self.length / self.batch_size))


def get_padding(in_size: Size, out_size: Size) -> Padding:
    d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return (p_h1, p_h2, p_w1, p_w2)


def get_loaders(batch_size: int,
                test_batch_size: int,
                use_cuda: bool,
                dataset: str = "FashionMNIST",
                in_size: Size = None,
                in_memory: bool = True):

    if in_size is None:
        in_size = (1, 32, 32)
    padding = get_padding(ORIGINAL_SIZE[dataset], in_size)
    mean, std, = MEAN_STD[dataset][in_size]

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trainset = getattr(datasets, dataset)(
        f'./.data/{dataset:s}',
        train=True, download=True,
        transform=transforms.Compose([
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.expand(in_size)),
            transforms.Normalize((mean,), (std,))
        ]))
    train_loader = DataLoader(
        trainset,
        batch_size=(len(trainset) if in_memory else batch_size), shuffle=True,
        **kwargs)
    test_loader = DataLoader(
        getattr(datasets, dataset)(
            f'./.data/{dataset:s}',
            train=False,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    if in_memory:
        train_loader = InMemoryDataLoader(train_loader, batch_size, shuffle=True)
        test_loader = InMemoryDataLoader(test_loader, test_batch_size)

    return train_loader, test_loader, (in_size, 10, (mean, std))
