import torch
import torch.nn as nn 
import torch.nn.functional as F

from model.modules import SpiralTransform, HarmonicStacking, SpiralEmbeddings
from model.modules import MLPHeadL, MLPHeadS
from model.modules import ConvEncoderBasic, ConvEncoderHarmonic

import matplotlib.pyplot as plt
import librosa


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input/output settings
        self.nframe      = config['input']['nframe']
        self.nbin        = config['spec']['n_bins']
        self.len_padding = config['input']['len_padding']
        self.npitch      = config['midi']['num_notes']

        # model settings
        self.cnn_kernel  = 5
        self.cnn_channel = 4

        # latent dims
        self.frame_dim = 8
        self.note_dim  = 88

        # useful
        self.len_frame     = 2 * self.len_padding + 1
        self.frame_cnn_dim = self.cnn_channel * (self.len_frame - (self.cnn_kernel - 1))
        self.freq_cnn_dim  = self.cnn_channel * (self.nbin - (self.cnn_kernel - 1))

        # model functions
        self.frame_conv   = nn.Conv2d(
            in_channels  = 1,
            out_channels = self.cnn_channel,
            kernel_size  = (1, self.cnn_kernel)
        )
        self.frame_linear = nn.Sequential(
            nn.Linear(self.frame_cnn_dim, self.frame_dim),
            nn.ReLU()
        )
        self.freq_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = self.cnn_channel,
            kernel_size  = (1, self.cnn_kernel)
        )
        self.freq_linear = nn.Sequential(
            nn.Linear(self.freq_cnn_dim, self.note_dim),
            nn.ReLU()
        )
        self.diameter_linear = nn.Sequential(
            nn.Linear(self.note_dim*self.frame_dim, 1),
        )
        self.center_linear = nn.Sequential(
            nn.Linear(self.note_dim*self.frame_dim, 3),
        )

    def chunk2frames(self, chunk: torch.Tensor):
        """ 
        x: Chunk [batch_size, nframe, nbin]
        y: Output tensors [batch_size, nframe, nbin, 2*len_padding+1]
        """
        pad_value = -80.0
        chunk = chunk.transpose(1, 2)
        chunk = F.pad(chunk, (self.len_padding, self.len_padding), 'constant', value=pad_value)
        chunk = chunk.transpose(1, 2)
        frames = chunk.unfold(dimension=1, size=self.len_frame, step=1)
        return frames
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # [B. nframe, nbin]
        x = self.chunk2frames(x)
        # [B, nframe, nbin, 2M+1]
        x = x.reshape(batch_size*self.nframe, self.nbin, self.len_frame).unsqueeze(1)
        # [B*nframe, 1, nbin, 2M+1]
        x = self.frame_conv(x).permute(0, 2, 1, 3).contiguous()
        # [B*nframe, nbin, cnn_channel, 2M+1-(cnn_kernel-1)]
        x = x.reshape(batch_size*self.nframe, self.nbin, self.frame_cnn_dim)
        # [B*nframe, nbin, frame_cnn_dim]
        x = self.frame_linear(x)
        # [B*nframe, nbin, frame_dim]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(1)
        # [B*nframe, 1, frame_dim, nbin]
        x = self.freq_conv(x).permute(0, 2, 1, 3).contiguous()
        # [B*nframe, frame_dim, cnn_channel, nbin-(cnn_kernel-1)]
        x = x.reshape(batch_size*self.nframe, self.frame_dim, self.freq_cnn_dim)
        # [B*nframe, frame_dim, freq_cnn_dim]
        x = self.freq_linear(x)
        # [B*nframe, frame_dim, note_dim]
        x = x.reshape(batch_size, self.nframe, self.frame_dim*self.note_dim)
        # [B, nframe, frame_dim*note_dim]
        spiral_cd = self.diameter_linear(x)
        # [B, nframe, 1]
        spiral_cc = self.center_linear(x)
        # [B, nframe, 3]
        return spiral_cd, spiral_cc
    

class SNAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input/output settings
        self.nframe      = config['input']['nframe']
        self.nbin        = 295
        self.len_padding = config['input']['len_padding']
        self.npitch      = config['midi']['num_notes']

        self.register_buffer('center_scale', torch.tensor([
            config['midr']['spiral']['radius'],  # x
            config['midr']['spiral']['radius'],  # y
            config['midr']['spiral']['height']   # z
        ], dtype=torch.float32))
        self.layernorm = nn.LayerNorm(normalized_shape=self.nbin)

        self.convenc = ConvEncoderBasic(ch_in=1, ch_out=1)

        self.flatten = nn.Flatten(start_dim=1)

        self.batchnorm = nn.BatchNorm1d(self.nbin)

        self.conv2spiral = SpiralTransform(
            config=config,
            freq_bins=self.nbin,
        )

        self.fc_center = MLPHeadL(
            dim_in=self.nbin, 
            dim_out=3, 
            activation='tanh')

        self.fc_diam = MLPHeadL(
            dim_in=self.nbin, 
            dim_out=1, 
            activation='softplus')
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # [B. nframe, nbin]
        x = self.layernorm(x)
        # [B, nframe, nbin]
        x = x.view(batch_size*self.nframe, 1, self.nbin)
        # [B*nframe, nbin]
        x = self.convenc(x).squeeze(1)
        # [B*nframe, nbin]
        spiral_cd = self.fc_diam(x)
        # [B*nframe, 1]
        spiral_cc = self.fc_center(x)
        # [B*nframe, 3]
        spiral_cd = spiral_cd.view(batch_size, self.nframe, 1)
        # [B, nframe, 1]
        spiral_cc = spiral_cc.view(batch_size, self.nframe, 3) * self.center_scale
        # [B, nframe, 3]
        return spiral_cd, spiral_cc
    
    

class Model2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input/output settings
        self.nframe      = config['input']['nframe']
        self.nbin        = 295
        self.len_padding = config['input']['len_padding']
        self.npitch      = config['midi']['num_notes']

        r = config['midr']['spiral']['radius']
        h = config['midr']['spiral']['height']
        harmonics = 8

        self.register_buffer('center_scale', torch.tensor([r, r, h], dtype=torch.float32))
        self.register_buffer('pe', SpiralEmbeddings(r, h).pe)

        self.layernorm = nn.LayerNorm(normalized_shape=self.nbin)
        self.convenc   = ConvEncoderHarmonic(harmonics)
        self.mlp       = MLPHeadS(88, 12, 'leaky_relu')
        self.diam_mlp  = MLPHeadS(12, 1, 'softplus')

    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        # plot_tensor(x.squeeze(0))
        # [B. nframe, nbin]
        x = self.layernorm(x)
        # [B, nframe, nbin]
        x = x.reshape(batch_size*self.nframe, self.nbin).unsqueeze(1)
        # [B*nframe, nbin]
        x = self.convenc(x).squeeze(1)
        # [B*nframe, 88]
        # plot_tensor(x)
        x = self.mlp(x)
        # [B*nframe, 12]
        # plot_tensor(x)
        spiral_cc = x @ self.pe
        # [B*nframe, 1]
        spiral_cd = self.diam_mlp(x)
        # [B*nframe, 3]
        spiral_cd = spiral_cd.reshape(batch_size, self.nframe, 1)
        # [B, nframe, 1]
        spiral_cc = spiral_cc.reshape(batch_size, self.nframe, 3)
        # [B, nframe, 3]
        return spiral_cd, spiral_cc
    


def plot_tensor(x):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        x.cpu().numpy().T,
        sr=16000,
        cmap="magma"
    )
    plt.title("Spec")
    plt.tight_layout()
    plt.show()