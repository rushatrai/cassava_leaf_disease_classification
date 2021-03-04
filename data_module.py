from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.dims = config['img_dims']
        self.batch_size = config['batch_size']

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize(self.dims),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        full_train_set = ImageFolder(
            root='/mnt/vol_b/datasets/cassava-leaf-disease-classification/train_images_sorted', transform=transform)
            
        val_pct = 0.2
        train_size = int((1-val_pct)*len(full_train_set))
        val_size = len(full_train_set) - train_size
        self.train_set, self.val_set = random_split(
            full_train_set, [train_size, val_size])

    def train_dataloader(self):
        train_set_loader = DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        return train_set_loader

    def val_dataloader(self):
        val_set_loader = DataLoader(
            dataset=self.val_set, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        return val_set_loader
