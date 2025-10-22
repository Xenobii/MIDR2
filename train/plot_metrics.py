import torch
import json
import glob
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def plot_loss(path):
    train_loss    = []
    train_loss_cd = []
    train_loss_cc = []
    valid_loss    = []
    valid_loss_cd = []
    valid_loss_cc = []
    
    for path in sorted(glob.glob(f'model/{path}/checkpoint_*.pth')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        train_loss.append(checkpoint['epoch_loss_train'])
        train_loss_cd.append(checkpoint['epoch_loss_train_cd'])
        train_loss_cc.append(checkpoint['epoch_loss_train_cc'])
        valid_loss.append(checkpoint['epoch_loss_valid'])
        valid_loss_cd.append(checkpoint['epoch_loss_valid_cd'])
        valid_loss_cc.append(checkpoint['epoch_loss_valid_cc'])

    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training and Validation Loss Components', fontsize=16)

    # ---- (1) Total Loss ----
    axes[0].plot(epochs, train_loss, label='Train', linewidth=2)
    axes[0].plot(epochs, valid_loss, label='Validation', linewidth=2)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # ---- (2) CD Loss ----
    axes[1].plot(epochs, train_loss_cd, label='Train', linewidth=2)
    axes[1].plot(epochs, valid_loss_cd, label='Validation', linewidth=2)
    axes[1].set_title('CD Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    # ---- (3) CC Loss ----
    axes[2].plot(epochs, train_loss_cc, label='Train', linewidth=2)
    axes[2].plot(epochs, valid_loss_cc, label='Validation', linewidth=2)
    axes[2].set_title('CC Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', default='model')
    args = parser.parse_args()
    path = args.path

    with open('config.json') as f:
        config = json.load(f)
    plot_loss(path)
    