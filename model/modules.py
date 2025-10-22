import math
import torch 
import torch.nn as nn
import torch.nn.functional as f

import warnings
warnings.filterwarnings(
    "ignore",
    message="Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy",
)



# CONVOLUTIONAL BLOCKS

class ConvEncoderBasic(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=20,
                kernel_size=15,
                padding=15//2,
                stride=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=20,
                out_channels=20,
                kernel_size=1,
                padding=0,
                stride=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(
                in_channels=20,
                out_channels=10,
                kernel_size=1,
                padding=0,
                stride=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(
                in_channels=10,
                out_channels=1,
                kernel_size=1,
                padding=0,
                stride=1
            ),
        )

    def forward(self, x):
        # [B, ch_in, nbin]
        x = self.conv1(x)
        # [B, 20, nbin]
        x = self.conv2(x)
        # [B, ch_out, nbin]
        return x


class ConvEncoderHarmonic(nn.Module):
    def __init__(self, harmonics):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=36,
                padding=2,
                stride=1
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.harmonicstacking = HarmonicStacking(harmonics)

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=harmonics,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=36,
                padding='same',
                stride=1
            ),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=12,
                padding=5,
                stride=3
            ),
            nn.LeakyReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=3//2,
                stride=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(
                in_channels=64,
                out_channels=1,
                kernel_size=36,
                padding='same',
                stride=1
            ),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        # [B, 295]
        x = self.conv1(x).squeeze(1)
        # [B, 264]
        x = self.harmonicstacking(x)
        # [B, 8, 264]
        x = self.conv2(x)
        # [B, 64, 264]
        x = self.conv3(x)
        # [B, 64, 88]
        x = self.conv4(x).squeeze(1)
        # [B, 88]
        return x


# MLP BLOCKS

class MLPHeadL(nn.Module):
    def __init__(self, dim_in, dim_out, activation='relu'):
        super().__init__()

        self.activatation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activatation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()

        self.fc = nn.Sequential(
            nn.Linear(dim_in, 88),
            nn.LeakyReLU(inplace=True),
            nn.Linear(88, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 88),
            nn.LeakyReLU(inplace=True),
            nn.Linear(88, 12),
            nn.LeakyReLU(inplace=True),
            nn.Linear(12, dim_out),
        )

    def forward(self, x):
        return self.activatation(self.fc(x))
        


class MLPHeadS(nn.Module):
    def __init__(self, dim_in, dim_out, activation='relu'):
        super().__init__()

        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()

        self.fc = nn.Sequential(
            nn.Linear(dim_in, 88),
            nn.LeakyReLU(inplace=True),
            nn.Linear(88, 12),
            nn.LeakyReLU(inplace=True),
            nn.Linear(12, dim_out),
        )

    def forward(self, x):
        return self.activation(self.fc(x))


# TRANSFORMS

class SpiralTransform(nn.Module):
    """
    Transform frequency-axis input [B, F] into [B, F, 3] spiral coordinates.
    The transformation is deterministic (based on geometric parameters).
    """
    def __init__(self, config, freq_bins):
        super().__init__()

        self.freq_bins = freq_bins

        self.r = config['midr']['spiral']['radius']
        self.h = config['midr']['spiral']['height']

        self.bins_per_semitone = config['spec']['bins_per_octave'] / 12
        self.freq_bins         = freq_bins

        freq_indices = torch.arange(freq_bins).float()
        theta = freq_indices * (math.pi / 2) / self.bins_per_semitone

        x_pos = self.r * torch.cos(theta)
        y_pos = self.r * torch.sin(theta)
        z_pos = self.h * theta / (12 * math.pi)

        pos = torch.stack([x_pos, y_pos, z_pos], dim=-1)  # [F, 3]
        self.register_buffer("spiral_coords", pos)

    def forward(self, x: torch.Tensor):
        """
        x: [B, F]
        returns: [B, F, 3]
        """
        B, F = x.shape
        if F != self.freq_bins:
            raise ValueError(f"Expected freq_bins={self.freq_bins}, got {F}")

        # [B, F, 1] × [1, F, 3] → [B, F, 3]
        spiral = x.unsqueeze(-1) * self.spiral_coords.unsqueeze(0)
        return spiral


class SpiralEmbeddings(nn.Module):
    def __init__(self, r, h):
        super().__init__()

        i = torch.arange(12, dtype=torch.float32)
        angles = i * (torch.pi / 2)
        angles = self.unwarp_angles(angles, T=6*torch.pi) 
        x = r * torch.cos(angles)
        y = r * torch.sin(angles)
        z = h * angles / (12*torch.pi)
        self.pe = torch.stack([x, y, z], dim=1)   # [12, 3]

    def unwarp_angles(self, angles: torch.Tensor, T=2*torch.pi, ref_angle=0):
        return (ref_angle + (angles - ref_angle + T/2) % T - T / 2)

    def forward(self):
        return self.pe
    


class HarmonicStacking(nn.Module):
    def __init__(self, harmonics: int = 8):
        super().__init__()
        self.H = harmonics
        self.shifts = [int(36 * math.log2(h + 2)) for h in range(harmonics)]
        self.max_shift = max(self.shifts)
        
    def forward(self, x: torch.Tensor):
        B, nbins = x.shape
        # [B, nbin]
        output = torch.zeros(B, self.H, nbins, device=x.device, dtype=x.dtype)
        # [B, H, nbin]
        for h, shift in enumerate(self.shifts):
            if shift < nbins:
                output[:, h, :nbins-shift] = x[:, shift:]
        # [B, H, nbin]
        return output