import json
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram


class SpecProcessor():
    def __init__(self, config):
        # feature
        self.sr             = config['feature']['sr']
        self.hop_sample     = config['feature']['hop_sample']
        self.fft_bins       = config['feature']['fft_bins']
        self.window_length  = config['feature']['window_length']
        self.log_offset     = config['feature']['log_offset']
        self.pad_mode       = config['feature']['pad_mode']
        self.mel_bins       = config['feature']['mel_bins']

    def wav2feat(self, f_wav):
        wave, sr = torchaudio.load(f_wav)
        wave_mono = torch.mean(wave, dim=0)
        tr_fsconv = Resample(sr, self.sr)
        wave_mono_16k = tr_fsconv(wave_mono)
        tr_mel = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.fft_bins, 
            win_length=self.window_length, 
            hop_length=self.hop_sample,
            pad_mode=self.pad_mode,
            n_mels=self.mel_bins,
            norm='slaney')
        mel_spec = tr_mel(wave_mono_16k)
        a_feature = (torch.log(mel_spec + self.log_offset)).T

        return a_feature