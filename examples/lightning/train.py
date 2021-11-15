# TODO: Install apex: https://github.com/NVIDIA/apex#quick-start


import os
import math
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
from LARC import LARC


# test model, a resnet 50

resnet = models.resnet50(pretrained=False)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 4096
EPOCHS     = 1000
BASE_LR    = 0.3
FINAL_LR   = 0
WEIGHT_DECAY = 1.5 * (10**-6)
NUM_GPUS   = 2 # TODO: Modify to actualy ##
NUM_PROCESSES = 1 # TODO: Modify to actual ##
IMAGE_SIZE = 224
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
WARMUP_EPOCHS = 5
START_WARMUP  = 0
ROOT_DIR= '.' # TODO: Modify this for root directory
NUM_WORKERS = torch.get_num_threads()

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=BASE_LR,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )
        self.optimizer = LARC(optimizer)
        warmup_lr_schedule = np.linspace(START_WARMUP, BASE_LR, len(train_loader) * WARMUP_EPOCHS)
        iters = np.arange(len(train_loader) * (EPOCHS - WARMUP_EPOCHS))
        cosine_lr_schedule = np.array([FINAL_LR + 0.5 * (BASE_LR - FINAL_LR) * (1 + \
                             math.cos(math.pi * t / (len(train_loader) * (EPOCHS - WARMUP_EPOCHS)))) for t in iters])
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        #return optimizer

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def optimizer_step(self, epoch, batch_idx, optimizer, **kwargs):
        iteration = epoch * len(train_loader) + batch_idx
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[iteration]
        self.optimizer.step()


# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# main

if __name__ == '__main__':
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        default_root_dir=ROOT_DIR,
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        accelerator='gpu',
        strategy='ddp',
        num_processes=NUM_PROCESSES,
        sync_batchnorm = True
    )

    trainer.fit(model, train_loader)
