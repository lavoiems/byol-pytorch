import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from pathlib import Path
from torchvision import transforms

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, transform):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in ['.jpg', '.png', '.jpeg']:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

class ImageNetDM(pl.LightningDataModule):
    def __init__(self, root_path, batch_size, num_workers, image_size=224):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])
        super().__init__(
            train_transforms=transform, val_transforms=transform, 
            test_transforms=transform)
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.train_dataset = ImagesDataset(os.path.join(self.root_path, 'train'), transform=self.train_transforms)
        self.val_dataset = ImagesDataset(os.path.join(self.root_path, 'val'), transform=self.train_transforms)
        self.test_dataset = self.val_dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
