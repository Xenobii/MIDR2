import math
import json
import torch
import torchaudio
import librosa
import numpy as np

import torch.nn.functional as F
import torchaudio.transforms as T
from nnAudio.features import CQT1992v2

import matplotlib.pyplot as plt
import librosa


import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

class WavPreprocessor():
    def __init__(self, config):
        # spec
        self.spec_type       = config['spec']['spec_type']
        self.sr              = config['spec']['sr']
        self.hop_length      = config['spec']['hop_sample']
        self.mel_bins        = config['spec']['mel_bins']
        self.n_bins          = config['spec']['n_bins']
        self.n_fft           = config['spec']['fft_bins']
        self.win_length      = config['spec']['window_length']
        self.log_offset      = config['spec']['log_offset']
        self.window          = config['spec']['window']
        self.pad_mode        = config['spec']['pad_mode']
        self.bins_per_octave = config['spec']['bins_per_octave']
        self.fmin            = config['spec']['fmin']
        self.fmax            = config['spec']['fmax']

        # input
        self.nframe      = config['input']['nframe']
        self.len_padding = config['input']['len_padding']

        # operations
        self.spec_transform = None
        if self.spec_type == "log-mel":
            self.spec_transform = T.MelSpectrogram(
                sample_rate = self.sr,
                n_fft       = self.n_fft,
                win_length  = self.win_length,
                hop_length  = self.hop_length,
                pad_mode    = self.pad_mode,
                n_mels      = self.mel_bins,
                norm        = 'slaney'
            )
        elif self.spec_type == "cqt":
            self.spec_transform = CQT1992v2(
                sr              = self.sr,
                hop_length      = self.hop_length,
                fmin            = self.fmin,
                fmax            = self.fmax,
                n_bins          = self.n_bins,
                bins_per_octave = self.bins_per_octave,
                window          = self.window,
                trainable       = False
            )

    @torch.inference_mode()
    def wav2spec(self, f_wav):
        # Load file
        wave, sr  = torchaudio.load(f_wav)
        resampler = T.Resample(sr, self.sr)
        wave = torch.mean(wave, dim=0)
        wave = resampler(wave)

        spec = self.spec_transform(wave)
        spec = torch.log(spec + self.log_offset).squeeze(0)

        return spec
    
    def spec2chunks(self, spec: torch.Tensor):
        pad_value = -80.0
        nframe    = self.nframe

        # 1. Pad right to make multiple of nframe
        num_frame_spec = spec.shape[1]
        num_chunks     = math.ceil(num_frame_spec / nframe)
        num_frame_pad  = num_chunks * nframe - num_frame_spec
        spec           = F.pad(spec, (0, num_frame_pad), mode='constant', value=pad_value)
        spec           = spec.transpose(0, 1)

        # 2. Split directly into non-overlapping chunks
        chunks = spec.unfold(dimension=0, size=nframe, step=nframe) 
        return chunks.permute(0, 2, 1).contiguous().numpy() # (num_chunks, nframe, n_mels)
    
    def __call__(self, x):
        x = self.wav2spec(x)
        x = self.spec2chunks(x)
        return x
    
    def plot_spec(self, spec):
        plt.figure(figsize=(10, 4))
        if isinstance(spec, torch.Tensor):
            librosa.display.specshow(
                spec.numpy(),
                sr=self.sr,
                cmap="magma"
            )
        elif isinstance(spec, np.ndarray):
            librosa.display.specshow(
                spec,
                sr=self.sr,
                cmap="magma"
            )
        plt.title("Spec")
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    preproc = WavPreprocessor(config)

    f_wav = "test_files/test_wav.WAV"
    spec = preproc.wav2spec(f_wav)
    chunks = preproc.spec2chunks(spec)
    # print(chunks.shape)

    preproc.plot_spec(spec)