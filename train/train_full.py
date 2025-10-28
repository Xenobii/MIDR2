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

from model.model import AMT_1
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
        self.model = AMT_1(config)
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

        self.criterion_cc       = nn.HuberLoss(reduction='mean')
        self.criterion_cd       = nn.HuberLoss(reduction='mean')
        self.criterion_mpe      = nn.BCELoss(reduction='mean')
        self.criterion_onset    = nn.BCELoss(reduction='mean')
        self.criterion_offset   = nn.BCELoss(reduction='mean')
        self.criterion_velocity = nn.CrossEntropyLoss()

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

            epoch_loss_train, \
            epoch_loss_train_mpe, \
            epoch_loss_train_onset, \
            epoch_loss_train_offset, \
            epoch_loss_train_velocity \
            = self.step_train()

            epoch_loss_valid, \
            epoch_loss_valid_mpe, \
            epoch_loss_valid_onset, \
            epoch_loss_valid_offset, \
            epoch_loss_valid_velocity \
            = self.step_valid()            
            
            t1 = time.time()
            print(f"Training time           : {(t1-t0):.3f} seconds")
            print(f"Training Loss           : {epoch_loss_train}")
            print(f"    Training Loss mpe       : {epoch_loss_train_mpe}")
            print(f"    Training Loss onset     : {epoch_loss_train_onset}")
            print(f"    Training Loss offset    : {epoch_loss_train_offset}")
            print(f"    Training Loss velocity  : {epoch_loss_train_velocity}")
            print(f"Validation Loss         : {epoch_loss_valid}")
            print(f"    Validation Loss mpe     : {epoch_loss_valid_mpe}")
            print(f"    Validation Loss onset   : {epoch_loss_valid_onset}")
            print(f"    Validation Loss offset  : {epoch_loss_valid_offset}")
            print(f"    Validation Loss velocity: {epoch_loss_valid_velocity}")

            # Save model
            torch.save({
                'epoch'                    : epoch,

                'epoch_loss_train'         : epoch_loss_train,
                'epoch_loss_train_mpe'     : epoch_loss_train_mpe,
                'epoch_loss_train_onset'   : epoch_loss_train_onset,
                'epoch_loss_train_offset'  : epoch_loss_train_offset,
                'epoch_loss_train_velocity': epoch_loss_train_velocity,

                'epoch_loss_valid'         : epoch_loss_valid,
                'epoch_loss_valid_mpe'     : epoch_loss_valid_mpe,
                'epoch_loss_valid_onset'   : epoch_loss_valid_onset,
                'epoch_loss_valid_offset'  : epoch_loss_valid_offset,
                'epoch_loss_valid_velocity': epoch_loss_valid_velocity,
                
                'optimizer_dict'           : self.optimizer.state_dict(),
                'model_dict'               : self.model.state_dict()
            }, f'model/{self.path}/checkpoint_{epoch}.pth')
            print(f"Checkpoint saved at: model/{self.path}/checkpoint_{epoch}.pth")

            self.scheduler.step(epoch_loss_valid)


    def step_train(self):
        self.model.train()
        epoch_loss = 0
        
        epoch_loss_mpe      = 0
        epoch_loss_onset    = 0
        epoch_loss_offset   = 0
        epoch_loss_velocity = 0

        epoch_loss_cc = 0
        epoch_loss_cd = 0

        eps = 1e-8

        pbar = tqdm(self.dataloader_train, desc="Training", leave=False)

        for spec, label_mpe, label_onset, label_offset, label_velocity, label_cd, label_cc, note_mask in pbar:
            spec = spec.to(self.device, non_blocking=True)

            label_mpe      = label_mpe.to(self.device, non_blocking=True)
            label_onset    = label_onset.to(self.device, non_blocking=True)
            label_offset   = label_offset.to(self.device, non_blocking=True)
            label_velocity = label_velocity.to(self.device, non_blocking=True)

            
            # label_cd = label_cd.to(self.device, non_blocking=True)
            # label_cc = label_cc.to(self.device, non_blocking=True)
            
            # note_mask = note_mask.to(dtype=torch.float32, device=self.device, non_blocking=True)
        
            self.optimizer.zero_grad(set_to_none=True)

            # AMP
            # with autocast(device_type=self.device, dtype=torch.float16):
            # Forward
            output_mpe, output_onset, output_offset, output_velocity = \
                self.model(spec)
            # Convert velocity to appropriate shape and dtype
            output_velocity = output_velocity.permute(0, 3, 1, 2).contiguous()
            
            # Calculate loss
            loss_mpe      = self.criterion_mpe(output_mpe, label_mpe)
            loss_onset    = self.criterion_onset(output_onset, label_onset)
            loss_offset   = self.criterion_offset(output_offset, label_offset)
            loss_velocity = self.criterion_velocity(output_velocity, label_velocity)

            # loss_cd = self.criterion_cd(label_cd, output_cd)
            # loss_cc = self.criterion_cc(label_cc, output_cc)
            # Apply mask
            # mask_cd = note_mask.expand_as(loss_cd)
            # mask_cc = note_mask.expand_as(loss_cc)
            # loss_cd = (loss_cd * mask_cd).sum() / (mask_cd.sum() + eps)
            # loss_cc = (loss_cc * mask_cc).sum() / (mask_cc.sum() + eps)
            
            # loss = loss_cd + loss_cc
            loss = loss_mpe + loss_onset + loss_offset + loss_velocity

            # Scale & Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Epoch loss
            epoch_loss += loss.item()

            epoch_loss_mpe      += loss_mpe.item()
            epoch_loss_onset    += loss_onset.item()
            epoch_loss_offset   += loss_offset.item()
            epoch_loss_velocity += loss_velocity.item()
            # epoch_loss_cd += loss_cd.item()
            # epoch_loss_cc += loss_cc.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        epoch_loss = epoch_loss / len(self.dataloader_train)
        # epoch_loss_cd = epoch_loss_cd / len(self.dataloader_train)
        # epoch_loss_cc = epoch_loss_cc / len(self.dataloader_train)
        return epoch_loss, epoch_loss_mpe, epoch_loss_onset, epoch_loss_offset, epoch_loss_velocity
    
    def step_valid(self):
        self.model.eval()
        epoch_loss = 0

        epoch_loss_mpe      = 0
        epoch_loss_onset    = 0
        epoch_loss_offset   = 0
        epoch_loss_velocity = 0

        epoch_loss_cc = 0
        epoch_loss_cd = 0

        eps = 1e-8
        
        with torch.no_grad(), autocast(device_type=self.device, dtype=torch.float16):
            for i, (spec, label_mpe, label_onset, label_offset, label_velocity, label_cd, label_cc, note_mask) in enumerate(self.dataloader_valid):
                spec     = spec.to(self.device, non_blocking=True)
                
                label_mpe      = label_mpe.to(self.device, non_blocking=True)
                label_onset    = label_onset.to(self.device, non_blocking=True)
                label_offset   = label_offset.to(self.device, non_blocking=True)
                label_velocity = label_velocity.to(self.device, non_blocking=True)

                # label_cd = label_cd.to(self.device, non_blocking=True)
                # label_cc = label_cc.to(self.device, non_blocking=True)

                # note_mask = note_mask.to(dtype=torch.float32, device=self.device, non_blocking=True)

                # Forward
                output_mpe, output_onset, output_offset, output_velocity = \
                    self.model(spec)
                
                output_velocity = output_velocity.permute(0, 3, 1, 2).contiguous()
                
                # Calculate loss
                loss_mpe      = self.criterion_mpe(output_mpe, label_mpe)
                loss_onset    = self.criterion_onset(output_onset, label_onset)
                loss_offset   = self.criterion_offset(output_offset, label_offset)
                loss_velocity = self.criterion_velocity(output_velocity, label_velocity)

                # loss_cd = self.criterion_cd(label_cd, output_cd)
                # loss_cc = self.criterion_cc(label_cc, output_cc)
                # Mask
                # mask_cd = note_mask.expand_as(loss_cd)
                # mask_cc = note_mask.expand_as(loss_cc)
                # loss_cd = (loss_cd * mask_cd).sum() / (mask_cd.sum() + eps)
                # loss_cc = (loss_cc * mask_cc).sum() / (mask_cc.sum() + eps)
                
                loss = loss_mpe + loss_onset + loss_offset + loss_velocity

                epoch_loss += loss.item()

                epoch_loss_mpe      += loss_mpe.item()
                epoch_loss_onset    += loss_onset.item()
                epoch_loss_offset   += loss_offset.item()
                epoch_loss_velocity += loss_velocity.item()

                # epoch_loss_cd += loss_cd
                # epoch_loss_cc += loss_cc

        epoch_loss = epoch_loss / len(self.dataloader_valid)
        # epoch_loss_cd = epoch_loss_cd / len(self.dataloader_valid)
        # epoch_loss_cc = epoch_loss_cc / len(self.dataloader_valid)
        return epoch_loss, epoch_loss_mpe, epoch_loss_onset, epoch_loss_offset, epoch_loss_velocity


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='model')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_out = "model/model.pkl"
    trainer = Trainer(config, model_out, args.path)
    trainer.train()