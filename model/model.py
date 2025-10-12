import torch

import torch.nn as nn 
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input/output settings
        self.nframe      = config['input']['nframe']
        self.nbin        = config['input']['nbin']
        self.len_padding = config['input']['len_padding']
        self.npitch      = config['output']['npitch']
        self.midr_dim    = 3

        # model settings
        self.cnn_kernel  = 3
        self.cnn_channel = 4
        self.z_dim       = 256

        # useful
        self.len_frame   = 2 * self.len_padding + 1
        self.cnn_dim     = self.cnn_channel * (self.len_frame - (self.cnn_kernel - 1))

        # model functions
        self.frame_conv   = nn.Conv2d(
            in_channels  = 1,
            out_channels = self.cnn_channel,
            kernel_size  = (1, self.cnn_kernel)
        )
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels  = self.cnn_dim,
                out_channels = 128,
                kernel_size  = 3,
                stride       = 2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = 128,
                out_channels = 64,
                kernel_size  = 3,
                stride       = 2
            ),
            nn.ReLU()
        )
        self.midr_decoder = nn.Sequential(
            nn.Linear(self.npitch, self.midr_dim),
            nn.ReLU()
        )

    @torch.inference_mode()
    def chunk2frames(self, chunk: torch.Tensor):
        """ 
        x: Chunk [batch_size, nframe, nbin]
        y: Output tensors [batch_size, nframe, nbin, 2*len_padding+1]
        """
        pad_value = -80.0
        chunk = F.pad(chunk, (self.len_padding, self.len_padding), 'constant', value=pad_value)

        frames = chunk.unfold(dimension=2, size=self.len_frame, step=1)
        return frames
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # [B, nframe, nbin] (?, 256, 256)
        x = self.chunk2frames(x)
        # [B, nframe, nbin, len_frame] (?, 256, 256, 33)
        x = x.reshape(batch_size*self.nframe, self.nbin, self.len_frame).unsqueeze(1)
        # [B*nframe, nbin, len_frame] (?*256, 1, 256, 33)
        x = self.frame_conv(x).permute(0, 2, 1, 3).contiguous()
        # [B*nframe, nbin, cnn_channel, len_frame - (cnn_kernel - 1)] (?*256, 256, 4, 31)
        x = x.reshape(batch_size*self.nframe, self.nbin, self.cnn_dim)
        # [B*nframe, nbin, cnn_dim] (?*256, 256, 116)
        x = self.freq_encoder(x)
        # [B*nframe, ] (?*256, 64, 64)
        x = x.view(x.shape[0], x.shape[1], -1)
        # [B*nframe, ] (?*256, 64*64)
        x = self.midr_decoder(x)
        # [B*nframe, midr_dims] (?*256, 4)
        x = x.reshape(batch_size, self.nframe, self.midr_dim).unsqueeze(1)
        # [B, nframe, midr_dims] (?, 256, 4)
        return x
