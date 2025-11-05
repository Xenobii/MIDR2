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
from torchinfo import summary

from model.model import AMT_1, AMT_Padded
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
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Training on {self.device}")

        # Model settings
        self.model = AMT_Padded(config)
        self.model = self.model.to(self.device)
        self.info = summary(self.model,
                            input_size=(1, 128, 295),
                            col_names=("input_size", "output_size", "num_params"),
                            device=self.device)
        print(str(self.info))
        
        # print(f"Trainable model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
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
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            fused=True if self.device=='cuda' else False
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=1,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        self.scaler = GradScaler(enabled=(self.device == 'cuda'))

        pos_weight = 2.0
        # self.criterion_cc       = nn.HuberLoss(reduction='mean')
        # self.criterion_cd       = nn.HuberLoss(reduction='mean')
        self.criterion_mpe      = nn.BCELoss( reduction='mean')
        self.criterion_onset    = nn.BCELoss( reduction='mean')
        self.criterion_offset   = nn.BCELoss( reduction='mean')
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
            pin_memory  = True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else False
        )
        self.dataloader_valid = DataLoader(
            dataset_valid,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else False
        )

        steps_per_epoch = len(self.dataloader_train) // 2
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        t1 = time.time()
        print(f"Initialized trainer in {(t1 - t0):.3f} seconds")

    def train(self):
        os.makedirs(f"model/{self.path}", exist_ok=True)

        with open(f"model/{self.path}/model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(self.info))

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
            print(f"Training Loss           : {epoch_loss_train:.6f}")
            print(f"    Training Loss mpe       : {epoch_loss_train_mpe:.6f}")
            print(f"    Training Loss onset     : {epoch_loss_train_onset:.6f}")
            print(f"    Training Loss offset    : {epoch_loss_train_offset:.6f}")
            print(f"    Training Loss velocity  : {epoch_loss_train_velocity:.6f}")
            print(f"Validation Loss         : {epoch_loss_valid:.6f}")
            print(f"    Validation Loss mpe     : {epoch_loss_valid_mpe:.6f}")
            print(f"    Validation Loss onset   : {epoch_loss_valid_onset:.6f}")
            print(f"    Validation Loss offset  : {epoch_loss_valid_offset:.6f}")
            print(f"    Validation Loss velocity: {epoch_loss_valid_velocity:.6f}")

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

        for batch_idx, (spec, label_mpe, label_onset, label_offset, label_velocity, _, _, _) in enumerate(pbar):
            spec = spec.to(self.device, non_blocking=True)

            label_mpe      = label_mpe.to(self.device, non_blocking=True)
            label_onset    = label_onset.to(self.device, non_blocking=True)
            label_offset   = label_offset.to(self.device, non_blocking=True)
            label_velocity = label_velocity.to(self.device, non_blocking=True)

            # label_cd = label_cd.to(self.device, non_blocking=True)
            # label_cc = label_cc.to(self.device, non_blocking=True)
            
            # note_mask = note_mask.to(dtype=torch.float32, device=self.device, non_blocking=True)
        
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
            if (batch_idx + 1) % 2 == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

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

        n_batches            = len(self.dataloader_train)
        epoch_loss          /= n_batches
        epoch_loss_mpe      /= n_batches
        epoch_loss_onset    /= n_batches
        epoch_loss_offset   /= n_batches
        epoch_loss_velocity /= n_batches
        # epoch_loss_cd = epoch_loss_cd / len(self.dataloader_train)
        # epoch_loss_cc = epoch_loss_cc / len(self.dataloader_train)
        return epoch_loss, epoch_loss_mpe, epoch_loss_onset, epoch_loss_offset, epoch_loss_velocity
    
    @torch.no_grad()
    def step_valid(self):
        self.model.eval()
        epoch_loss = 0

        epoch_loss_mpe      = 0
        epoch_loss_onset    = 0
        epoch_loss_offset   = 0
        epoch_loss_velocity = 0

        # epoch_loss_cc = 0
        # epoch_loss_cd = 0

        # eps = 1e-8
        
        pbar = tqdm(self.dataloader_valid, desc="Validation", leave=False)
        for spec, label_mpe, label_onset, label_offset, label_velocity, _, _, _ in pbar:
            spec = spec.to(self.device, non_blocking=True)
            
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

            pbar.set_postfix({"loss": f"{epoch_loss / (pbar.n + 1):.4f}"})
        
        pbar.close()

        n_batches            = len(self.dataloader_valid)
        epoch_loss          /= n_batches
        epoch_loss_mpe      /= n_batches
        epoch_loss_onset    /= n_batches
        epoch_loss_offset   /= n_batches
        epoch_loss_velocity /= n_batches
        
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