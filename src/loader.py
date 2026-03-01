import torch 
from torch.utils.data import DataLoader


def train_load(train_set):
    train_loader = DataLoader(train_set,32,True)
    return train_loader

def test_load(test_set):
    test_loader = DataLoader(test_set,32,True)
    return test_loader