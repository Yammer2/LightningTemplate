import torch
from torch.utils.data import DataLoader, random_split

import lightning.pytorch as pl
from torchvision import transforms
from torchvision.datasets import CIFAR10

class CIFAR10_pl(pl.LightningDataModule):

    def __init__(self,
                 batch_size: int,
                 train_val_ratio: float = 0.8,
                 download: bool = True,
                 ) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.download = download


    def setup(self, stage: str) -> None:
        """ ==datasetの準備==
        train, valの時はstage='fit'
        testの時はstage='test'
        """

        # Transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        ########################################
        # Train and validation datasets
        ########################################
        if stage == 'fit' or stage is None:
            dataset = CIFAR10(root='../dataset', train=True, download=self.download, transform=transform)
            self.trainset, self.valset = random_split(dataset, [self.train_val_ratio, 1-self.train_val_ratio])

            print(f'len(trainset): {len(self.trainset)}')
            print(f'len(valset): {len(self.valset)}')

        ########################################
        # Test dataset
        ########################################
        elif stage == 'test' or stage is None:
            self.testset = CIFAR10(root='../dataset', train=False, download=True, transform=transform)
            print(f'len(testset): {len(self.testset)}')


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8,
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8,
                          )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8,
                          )
