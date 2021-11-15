# TODO: Install apex: https://github.com/NVIDIA/apex#quick-start


import os
import math
import argparse
import numpy as np

import torch
from torchvision import models

from byol_pytorch import BYOL
import pytorch_lightning as pl
from LARC import LARC
from imagenet import ImageNetDM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def get_lr_scheduler(start_warmup, base_lr, iter_per_epoch, epochs, final_lr, warmup_epochs):
    warmup_steps = iter_per_epoch * warmup_epochs
    total_steps = iter_per_epoch * epochs
    def _get_lr(global_step):
        if global_step < warmup_steps:
            return start_warmup + (base_lr - start_warmup) * global_step / warmup_steps
        elif global_step < total_steps:
            t = global_step - warmup_steps
            return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / (iter_per_epoch * (epochs - warmup_epochs))))
        else:
            return final_lr
    return _get_lr


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        resnet = models.resnet50(pretrained=False)
        self.learner = BYOL(
            net=resnet, image_size=224, hidden_layer='avgpool', 
             projection_size = 256,
            projection_hidden_size = 4096,
            moving_average_decay = 0.99)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log('train/loss', loss)
        self.log('train/lr', self.lr_schedule(self.trainer.global_step))
        return {'loss': loss}

    def configure_optimizers(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.base_lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        optimizer = LARC(optimizer)
        self.lr_schedule = get_lr_scheduler(
            start_warmup=self.hparams.start_warmup, 
            base_lr=self.hparams.base_lr, 
            final_lr=self.hparams.final_lr,
            iter_per_epoch=len(train_loader),
            epochs=self.hparams.epochs,
            warmup_epochs=self.hparams.warmup_epochs)
        return optimizer

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()
    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer= None, optimizer_idx: int = None, optimizer_closure = None, on_tpu: bool = None, using_native_amp: bool = None, using_lbfgs: bool = None) -> None:
        iteration = self.trainer.global_step
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule(iteration)
        return super().optimizer_step(epoch=epoch, batch_idx=batch_idx, optimizer=optimizer, optimizer_idx=optimizer_idx, optimizer_closure=optimizer_closure, on_tpu=on_tpu, using_native_amp=using_native_amp, using_lbfgs=using_lbfgs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--root_path', type=str, required = True,
                       help='path to your imagenet root')
        parser.add_argument('--base_lr', default=0.3, type=float)
        parser.add_argument('--final_lr', default=0, type=float)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--start_warmup', default=0, type=float)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--weight_decay', default=1.5 * (10**-6), type=float)
        parser.add_argument('--num_workers', default=12, type=int)
        parser.add_argument('--exp_name', default='byol', help='experiment results, logs, ckpt')
        return parser

# main
def main():
    # arguments
    parser = argparse.ArgumentParser(description='byol-lightning-test')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SelfSupervisedLearner.add_argparse_args(parser)
    args = parser.parse_args()
    model = SelfSupervisedLearner(**vars(args))
    dm = ImageNetDM(
        args.root_path, 
        image_size=224, batch_size=args.batch_size, 
        num_workers=args.num_workers)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
    )
    logger = TensorBoardLogger(
        'byol_lightning',
        name=args.exp_name,
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        progress_bar_refresh_rate=10,
        accelerator='ddp',
        sync_batchnorm=True,
        precision=args.precision,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        logger=[logger],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()