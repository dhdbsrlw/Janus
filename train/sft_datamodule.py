import os

from torch.utils.data import DataLoader
from dataset import T2ICompBench

import pytorch_lightning as pl


class SFTDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        category_list = ["complex"] # 오버피팅 테스트 목적
        self.train_dataset = T2ICompBench(self.config.train.data_dir,
                                          category_list=category_list,
                                          split="train")
        # self.val_dataset = None


    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                    dataset=self.train_dataset, 
                    shuffle=True,
                    collate_fn=self.train_dataset.collate_fn, # lambda batch: batch, 
                    batch_size=self.config.dataset.per_device_train_batch_size,  # Batch size per process
                    num_workers=self.config.dataset.preprocessing_num_workers,   # Number of data loading workers
                    pin_memory=True, 
                    drop_last=True                                               # Drop the last incomplete batch (important for DDP)
                    )


    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                    dataset=self.val_dataset, 
                    shuffle=False,
                    collate_fn=self.val_dataset.collate_fn, # lambda batch: batch, # TODO
                    batch_size=self.config.dataset.per_device_eval_batch_size,  # Batch size per process
                    num_workers=self.config.dataset.preprocessing_num_workers,   # Number of data loading workers
                    pin_memory=True, 
                    drop_last=True                                               # Drop the last incomplete batch (important for DDP)
                    )