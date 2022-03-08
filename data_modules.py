import os
from argparse import Namespace
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_path, shuffle=True, val_split=0.2, batch_size=256):
        super().__init__()
        self.data_dir = data_path
        self.train_transfs, self.test_transfs = get_transform({"dataset":"cifar10"})
        self.shuffle = shuffle
        self.val_split = val_split
        self.batch_size = batch_size
        try:
            self.num_workers = min(len(os.sched_getaffinity(0)*6),16)
        except:
            self.num_workers = 10
    
    def prepare_data(self):
        # Downloading data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.val_split > 0 :
                CIFAR10_full = CIFAR10(self.data_dir, train=True, transform=self.train_transfs)
                total_elem = len(CIFAR10_full)
                val_elem = int(total_elem*self.val_split)
                train_elem = total_elem - val_elem
                self.CIFAR10_train, self.CIFAR10_val = random_split(CIFAR10_full,[train_elem, val_elem])
            else :
                self.CIFAR10_train = CIFAR10(self.data_dir, train=True, transform=self.train_transfs)
        if stage == 'test' or stage is None:
            self.CIFAR10_test = CIFAR10(self.data_dir, train=False, transform=self.test_transfs)

    def train_dataloader(self):
        return DataLoader(self.CIFAR10_train, batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_split > 0:
            return DataLoader(self.CIFAR10_val, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
        else:
            return None

    def test_dataloader(self):
        return DataLoader(self.CIFAR10_test, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

def get_transform(config):

    normalization_values = {
        'cifar10':{
            'mean':[0.4914, 0.4822, 0.4465],
            'std':[0.2023, 0.1994, 0.2010]
        },
    }

    n_values = normalization_values[config['dataset']]

    print("Using CIFAR transform")

    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(n_values['mean'], n_values['std'])])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize(n_values['mean'], n_values['std'])])

    if config.get("no_data_augment", False):
        print("Warning: No data augmentation used !")
        return transforms.ToTensor(), transforms.ToTensor()
    else:
        return train_transform, test_transform

def get_data_module(config):

    if config["dataset"] == "cifar10":
        dm = CIFAR10DataModule(
            config["data_path"],
            shuffle=True,
            val_split=config["val_split"],
            batch_size=config["batch_size"])

        train_tr, test_tr = get_transform(config)

        dm.train_transfs = train_tr
        dm.val_transfs = test_tr
        dm.test_transfs = test_tr
    
        return dm

    else:
        return None