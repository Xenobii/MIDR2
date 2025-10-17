import os
import time
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import Model
from dataset.maestro import MaestroDataset



class Trainer():
    def __init__(self, config, model_out):
        print(f"** Training model **")
        t0 = time.time()
        
        self.model_out   = model_out
        self.lr          = config["training"]["lr"]
        self.batch_size  = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]
        self.epochs      = config["training"]["epochs"]

        # Torch settings
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Training on {self.device}")

        # Model settings
        self.model = Model(config)
        self.model = self.model.to(self.device)
        print(f"Trainable model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Training settings
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        self.criterion_spiral_cc = nn.L1Loss()
        self.criterion_spiral_cd = nn.L1Loss()

        # Dataset loading
        f_dataset = 'dataset/processed_dataset.h5'
        dataset_train = MaestroDataset(f_dataset, split='train')
        dataset_valid = MaestroDataset(f_dataset, split='validation')
        
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True
        )
        self.dataloader_valid = DataLoader(
            dataset_valid,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True
        )

        t1 = time.time()
        print(f"Initialized trainer in {(t1 - t0):.3f} seconds")

    def train(self):
        os.makedirs("model/checkpoints", exist_ok=True)
        for epoch in range(self.epochs):
            t0 = time.time()
            epoch_loss_train = self.step_train()
            epoch_loss_valid = self.step_valid()
            t1 = time.time()
            print(f"-- Epoch {epoch} --")
            print(f"Training time: {(t1-t0):.3f} seconds")
            print(f"Training Loss: {epoch_loss_train}")
            print(f"Validation Loss: {epoch_loss_valid}")

            # Save model
            torch.save({
                'epoch'           : epoch,
                'epoch_loss_train': epoch_loss_train,
                'epoch_loss_valid': epoch_loss_valid,
                'optimizer_dict'  : self.optimizer.state_dict(),
                'model_dict'      : self.model.state_dict()
            }, f'model/checkpoints/checkpoint{epoch}.pth')
            print(f"Checkpoint saved at: model/checkpoints/checkpoint{epoch}.pth")

            self.scheduler.step(epoch_loss_valid)


    def step_train(self):
        self.model.train()
        epoch_loss = 0

        pbar = tqdm(self.dataloader_train, desc="Training", leave=False)

        for spec, output_spiral_cd, output_spiral_cc in pbar:
            spec            = spec.to(self.device, non_blocking=True)
            label_spiral_cd = output_spiral_cd.to(self.device, non_blocking=True)
            label_spiral_cc = output_spiral_cc.to(self.device, non_blocking=True)
        
            self.optimizer.zero_grad()
            output_spiral_cd, output_spiral_cc = self.model(spec)

            # flatten
            # label_spiral_cd  = label_spiral_cd.contiguous().view(-1)
            # label_spiral_cc  = label_spiral_cc.contiguous().view(-1)
            # output_spiral_cd = output_spiral_cd.contiguous().view(-1)
            # output_spiral_cc = output_spiral_cc.contiguous().view(-1)

            loss_spiral_cd = self.criterion_spiral_cd(label_spiral_cd, output_spiral_cd)
            loss_spiral_cc = self.criterion_spiral_cc(label_spiral_cc, output_spiral_cc)
            loss = loss_spiral_cd + loss_spiral_cc

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

        pbar.close()
        return epoch_loss / len(self.dataloader_train)
    
    def step_valid(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for i, (spec, output_spiral_cd, output_spiral_cc) in enumerate(self.dataloader_valid):
                spec            = spec.to(self.device, non_blocking=True)
                label_spiral_cd = output_spiral_cd.to(self.device, non_blocking=True)
                label_spiral_cc = output_spiral_cc.to(self.device, non_blocking=True)
            
                # self.optimizer.zero_grad()
                output_spiral_cd, output_spiral_cc = self.model(spec)

                # flatten
                # label_spiral_cd  = label_spiral_cd.contiguous().view(-1)
                # label_spiral_cc  = label_spiral_cc.contiguous().view(-1)
                # output_spiral_cd = output_spiral_cd.contiguous().view(-1)
                # output_spiral_cc = output_spiral_cc.contiguous().view(-1)

                loss_spiral_cd = self.criterion_spiral_cd(label_spiral_cd, output_spiral_cd)
                loss_spiral_cc = self.criterion_spiral_cc(label_spiral_cc, output_spiral_cc)
                loss = loss_spiral_cd + loss_spiral_cc

                epoch_loss += loss.item()

        return epoch_loss / len(self.dataloader_valid)


if __name__=="__main__":
    with open('config.json') as f:
        config = json.load(f)
    model_out = "model/model.pkl"
    trainer = Trainer(config, model_out)
    trainer.train()