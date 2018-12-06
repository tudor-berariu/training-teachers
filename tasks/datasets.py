from math import ceil
from typing import Dict, Iterator, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tasks.random_dataset import RandomDataset
from numpy.random import permutation


# TODO: Other datasets than FashionMNIST


ORIGINAL_SIZE = {"FashionMNIST": (1, 28, 28)}
NCLASSES = {"FashionMNIST": 10,'random': 10}
MEAN_STD = {"FashionMNIST": {(1, 32, 32): (0.2190, 0.3318,),
               (1, 28, 28): (0.2860, 0.3530,)}}


Size = Tuple[int, int, int]
Padding = Tuple[int, int, int, int]


class InMemoryDataLoader(Iterator):
  # Only for small datasets

  def __init__(self, loader: DataLoader,
         batch_size: int,
         shuffle: bool = False,
         limit: int = 0) -> None:
    full_data, full_target = None, None
    self.batch_size = batch_size
    self.shuffle = shuffle
    for data, target in loader:
      if full_data is not None:
        full_data = torch.cat((full_data, data), dim=0)
        full_target = torch.cat((full_target, target), dim=0)
      else:
        full_data, full_target = data, target
    if limit==0 or limit>=len(self.data):
      self.data = full_data
      self.target = full_target
    else:
      perm = permutation(len(data))
      self.data = full_data[perm]
      self.target = full_target[perm]  
    self.length = len(self.data)
    self.idxs, self.offset = None, None

  # pylint: disable=invalid-name
  def to(self, device: torch.device) -> None:
    self.data, self.target = self.data.to(device), self.target.to(device)

  def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
    if self.shuffle:
      self.idxs = torch.randperm(self.length)
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
      return self.data[idxs], self.target[idxs], idxs

    return self.data[start:end], self.target[start:end], \
            torch.arange(start,end,dtype=torch.long)

  @property
  def ds_size(self):
    return len(self.data)

  def __len__(self) -> int:
    return int(ceil(self.length / self.batch_size))

  def sample(self, nsamples: int) -> Dict[str, Tensor]:
    if nsamples > self.length:
      raise ValueError("You ask for too much.")
    idxs = torch.randint(self.length, (nsamples,),
               dtype=torch.long, device=self.data.device)
    return {"data": self.data.index_select(0, idxs),
        "target": self.target.index_select(0, idxs)}


def get_padding(in_size: Size, out_size: Size) -> Padding:
  d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
  p_h1, p_w1 = d_h // 2, d_w // 2
  p_h2, p_w2 = d_h - p_h1, d_w - p_w1
  return p_h1, p_h2, p_w1, p_w2


def get_loaders(dataset: str,
        batch_size: int,
        test_batch_size: int,
        in_size: Size = None,
        normalize: bool = False,
        limit: int = 0):

  if dataset!='random':
    if in_size is None:
      in_size = (1, 32, 32)
    padding = get_padding(ORIGINAL_SIZE[dataset], in_size)

    transfs = [
      transforms.Pad(padding),
      transforms.ToTensor(),
      transforms.Lambda(lambda t: t.expand(in_size))
    ]

    if normalize:
      mean, std = nrmlz = MEAN_STD[dataset][in_size]
      transfs.append(transforms.Normalize((mean,), (std,)))
    else:
      nrmlz = None

    trainset = getattr(datasets, dataset)(
      f'./.data/{dataset:s}',
      train=True, download=True,
      transform=transforms.Compose(transfs))
  else:
    trainset = RandomDataset(40,NCLASSES[dataset], in_size)
    testset = trainset
    nrmlz=None
    
  train_loader = DataLoader(trainset, batch_size=len(trainset))

  if dataset!='random':
    testset = getattr(datasets, dataset)(
      f'./.data/{dataset:s}',
      train=False,
      transform=transforms.Compose(transfs))
  test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)

  train_loader = InMemoryDataLoader(train_loader, batch_size, shuffle=True,
                                    limit=limit)
  test_loader = InMemoryDataLoader(test_loader, test_batch_size, limit=limit)

  return train_loader, test_loader, (in_size, NCLASSES[dataset], nrmlz)
