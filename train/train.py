import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler

from model.model import Model, SNAModel, Model2
from dataset.maestro import MaestroDataset



class Trainer():
    def __init__(self, config, model_out, path):
        print(f"** Training model **")
        t0 = time.time()
        
        self.model_out   = model_out
        self.path        = path
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
        self.model = Model2(config)
        self.model = self.model.to(self.device)
        print(f"Trainable model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        # def count_parameters(model):
        #     total_params = 0
        #     for name, parameter in model.named_parameters():
        #         if not parameter.requires_grad:
        #             continue
        #         params = parameter.numel()
        #         print(f"{name}: {params}")
        #         total_params += params
        #     print(f"Total Trainable Params: {total_params}")
        #     return total_params
        
        # count_parameters(self.model)

        # Training settings
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.scaler    = GradScaler(enabled=(self.device == 'cuda'))

        self.criterion_spiral_cc = nn.HuberLoss(reduction='none')
        self.criterion_spiral_cd = nn.HuberLoss(reduction='none')

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
        os.makedirs(f"model/{self.path}", exist_ok=True)
        for epoch in range(self.epochs):
            print(f"-- Epoch {epoch} --")
            t0 = time.time()
            epoch_loss_train, epoch_loss_train_cd, epoch_loss_train_cc = self.step_train()
            epoch_loss_valid, epoch_loss_valid_cd, epoch_loss_valid_cc = self.step_valid()
            t1 = time.time()
            print(f"Training time           : {(t1-t0):.3f} seconds")
            print(f"Training Loss           : {epoch_loss_train}")
            print(f"    Training Loss CD    : {epoch_loss_train_cd}")
            print(f"    Training Loss CC    : {epoch_loss_train_cc}")
            print(f"Validation Loss         : {epoch_loss_valid}")
            print(f"    Validation Loss CD  : {epoch_loss_valid_cd}")
            print(f"    Validation Loss CC  : {epoch_loss_valid_cc}")

            # Save model
            torch.save({
                'epoch'              : epoch,
                'epoch_loss_train'   : epoch_loss_train,
                'epoch_loss_train_cd': epoch_loss_train_cd,
                'epoch_loss_train_cc': epoch_loss_train_cc,
                'epoch_loss_valid'   : epoch_loss_valid,
                'epoch_loss_valid_cd': epoch_loss_valid_cd,
                'epoch_loss_valid_cc': epoch_loss_valid_cc,
                'optimizer_dict'     : self.optimizer.state_dict(),
                'model_dict'         : self.model.state_dict()
            }, f'model/{self.path}/checkpoint_{epoch}.pth')
            print(f"Checkpoint saved at: model/{self.path}/checkpoint_{epoch}.pth")

            self.scheduler.step(epoch_loss_valid)


    def step_train(self):
        self.model.train()
        epoch_loss = 0
        epoch_loss_cc = 0 
        epoch_loss_cd = 0 
        eps = 1e-8

        pbar = tqdm(self.dataloader_train, desc="Training", leave=False)

        for spec, label_spiral_cd, label_spiral_cc, note_mask in pbar:
            spec            = spec.to(self.device, non_blocking=True)
            label_spiral_cd = label_spiral_cd.to(self.device, non_blocking=True)
            label_spiral_cc = label_spiral_cc.to(self.device, non_blocking=True)
            
            note_mask = note_mask.to(dtype=torch.float32, device=self.device, non_blocking=True)
        
            self.optimizer.zero_grad(set_to_none=True)

            # AMP
            with autocast(device_type=self.device, dtype=torch.float16):
                # Forward
                output_spiral_cd, output_spiral_cc = self.model(spec)
                
                # Calculate loss
                loss_spiral_cd = self.criterion_spiral_cd(label_spiral_cd, output_spiral_cd)
                loss_spiral_cc = self.criterion_spiral_cc(label_spiral_cc, output_spiral_cc)
                
                # Apply mask
                mask_cd = note_mask.expand_as(loss_spiral_cd)
                mask_cc = note_mask.expand_as(loss_spiral_cc)
                loss_spiral_cd = (loss_spiral_cd * mask_cd).sum() / (mask_cd.sum() + eps)
                loss_spiral_cc = (loss_spiral_cc * mask_cc).sum() / (mask_cc.sum() + eps)
                
                loss = loss_spiral_cd + loss_spiral_cc

            # Scale & Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Epoch loss
            epoch_loss    += loss.item()
            epoch_loss_cd += loss_spiral_cd.item()
            epoch_loss_cc += loss_spiral_cc.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        epoch_loss = epoch_loss / len(self.dataloader_train)
        epoch_loss_cd = epoch_loss_cd / len(self.dataloader_train)
        epoch_loss_cc = epoch_loss_cc / len(self.dataloader_train)
        return epoch_loss, epoch_loss_cd, epoch_loss_cc
    
    def step_valid(self):
        self.model.eval()
        epoch_loss    = 0
        epoch_loss_cc = 0
        epoch_loss_cd = 0

        eps = 1e-8
        
        with torch.no_grad(), autocast(device_type=self.device, dtype=torch.float16):
            for i, (spec, label_spiral_cd, label_spiral_cc, note_mask) in enumerate(self.dataloader_valid):
                spec            = spec.to(self.device, non_blocking=True)
                label_spiral_cd = label_spiral_cd.to(self.device, non_blocking=True)
                label_spiral_cc = label_spiral_cc.to(self.device, non_blocking=True)
                
                note_mask = note_mask.to(dtype=torch.float32, device=self.device, non_blocking=True)

                # Forward
                output_spiral_cd, output_spiral_cc = self.model(spec)
                
                # Calculate Loss
                loss_spiral_cd = self.criterion_spiral_cd(label_spiral_cd, output_spiral_cd)
                loss_spiral_cc = self.criterion_spiral_cc(label_spiral_cc, output_spiral_cc)

                # Mask
                mask_cd = note_mask.expand_as(loss_spiral_cd)
                mask_cc = note_mask.expand_as(loss_spiral_cc)
                loss_spiral_cd = (loss_spiral_cd * mask_cd).sum() / (mask_cd.sum() + eps)
                loss_spiral_cc = (loss_spiral_cc * mask_cc).sum() / (mask_cc.sum() + eps)
                
                loss = loss_spiral_cd + loss_spiral_cc

                epoch_loss    += loss.item()
                epoch_loss_cd += loss_spiral_cd
                epoch_loss_cc += loss_spiral_cc

        epoch_loss = epoch_loss / len(self.dataloader_valid)
        epoch_loss_cd = epoch_loss_cd / len(self.dataloader_valid)
        epoch_loss_cc = epoch_loss_cc / len(self.dataloader_valid)
        return epoch_loss, epoch_loss_cd, epoch_loss_cc


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='model')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_out = "model/model.pkl"
    trainer = Trainer(config, model_out, args.path)
    trainer.train()