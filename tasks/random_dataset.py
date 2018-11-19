import torch
from torch.utils.data import Dataset
import numpy as np

class RandomDataset(Dataset):

  def __init__(self, size, nclasses, shape = [32,32,1]):
    data_elems = int(size*np.prod(shape))
    self.data = torch.normal(torch.zeros(data_elems), torch.ones(data_elems))
    self.data = self.data.reshape([size,*shape])
    self.targets =  torch.randint(0,nclasses,[size],dtype=torch.long)

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    return self.data[index], self.targets[index]


if __name__=='__main__':
  #test
  ds = RandomDataset(10,100)
  print(ds[-1][1])