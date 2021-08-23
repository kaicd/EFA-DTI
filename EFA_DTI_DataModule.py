from typing import Callable

import dgl
import pytorch_lightning as pl
import torch as th
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from Utility.Dataset import EFA_DTI_Dataset


class EFA_DTI_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        reset: bool = False,
        y_transform: Callable = None,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.reset = reset
        self.y_transform = y_transform
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.dataset = EFA_DTI_Dataset(
            data_dir=self.data_dir,
            data_name=self.data_name,
            reset=self.reset,
            y_transform=self.y_transform,
        )
        self.train_idx, self.valid_idx = train_test_split(
            range(len(self.dataset)), test_size=0.1, random_state=self.seed
        )

    @staticmethod
    def dgl_collate(batch):
        g, fp, pt, y = zip(*batch)
        return (
            dgl.batch(g),
            th.cat(fp),
            th.cat(pt),
            th.cat(y),
        )

    def dataloader(self, split, shuffle):
        splits = {"train": self.train_idx, "valid": self.valid_idx}
        dataset = Subset(self.dataset, splits[split])

        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.dgl_collate,
            num_workers=4,
            pin_memory=True,
            **self.kwargs
        )
        return dl

    def train_dataloader(self, split="train", shuffle=True):
        return self.dataloader(split="train", shuffle=True)

    def val_dataloader(self, split="valid", shuffle=False):
        return self.dataloader(split="valid", shuffle=False)
