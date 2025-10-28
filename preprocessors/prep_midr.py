import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import json
import torch
import math
import torch.nn.functional as F

from midr.midr import Circle, Spiral


class MidrPreprocessor:
    def __init__(self, config):
        # spec
        self.fs = config['spec']['sr'] / config['spec']['hop_sample']
        
        # input
        self.nframe      = config['input']['nframe']
        self.len_padding = config['input']['len_padding']

        self.circle = Circle(config)
        self.spiral = Spiral(config)

    def midi2chroma(self, f_midi):
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        assert len(midi.instruments) == 1, f"Model is finetuned for single instrument"
        
        chroma = midi.instruments[0].get_chroma(fs=self.fs, pedal_threshold=120)
        
        # arrange chroma in circle of fifths
        circle_of_fifths = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5])
        chroma = chroma[circle_of_fifths]
        
        return chroma.T
    
    def chroma2midr(self, chroma):
        nframe      = len(chroma)
        a_circle_cc = np.zeros((nframe, 2), dtype=np.float32)
        a_circle_cd = np.zeros((nframe, 1), dtype=np.float32)
        a_spiral_cc = np.zeros((nframe, 3), dtype=np.float32)
        a_spiral_cd = np.zeros((nframe, 1), dtype=np.float32)
        a_note_mask = np.zeros((nframe, 1), dtype=np.bool)

        for i in range(nframe):
            pitches = np.nonzero(chroma[i])[0]
            if pitches.size > 0:
                velocities = chroma[i][pitches]
                # circle
                self.circle.create_cloud(pitches, velocities)
                a_circle_cc[i] = self.circle.center
                a_circle_cd[i] = self.circle.diameter
                # spiral
                self.spiral.create_cloud(pitches, velocities)
                a_spiral_cc[i] = self.spiral.center
                a_spiral_cd[i] = self.spiral.diameter

        # Convert to chunks
        a_circle_cc = self.midr2chunks(torch.from_numpy(a_circle_cc))
        a_circle_cd = self.midr2chunks(torch.from_numpy(a_circle_cd))
        a_spiral_cc = self.midr2chunks(torch.from_numpy(a_spiral_cc))
        a_spiral_cd = self.midr2chunks(torch.from_numpy(a_spiral_cd))
        a_note_mask = a_spiral_cd != 0

        a_midr = {
            'circle_cc' : a_circle_cc,
            'circle_cd' : a_circle_cd,
            'spiral_cc' : a_spiral_cc,
            'spiral_cd' : a_spiral_cd,
            'note_mask' : a_note_mask
        }
        return a_midr
    
    def midr2chunks(self, midr: torch.Tensor) -> torch.Tensor:
        pad_value = 0.0
        nframe    = self.nframe

        # Pad right to make multiple of nframe
        num_frame_midr = midr.shape[0]
        num_chunks     = math.ceil(num_frame_midr / nframe)
        num_frame_pad  = num_chunks * nframe - num_frame_midr
        midr           = F.pad(midr, (0, 0, 0, num_frame_pad), mode='constant', value=pad_value)

        # Split directly into non overlapping chunks
        chunks = midr.unfold(dimension=0, size=nframe, step=nframe) # (num_chunks, nframe, x)
        return chunks.permute(0, 2, 1).contiguous().numpy()
    
    def __call__(self, x):
        x = self.midi2chroma(x)
        x = self.chroma2midr(x)
        return x
    
    def plot_chroma(self, chroma):
        chroma = np.array(chroma)
        if chroma.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {chroma.shape}")

        plt.figure(figsize=(10, 4))
        if isinstance(chroma, np.ndarray):
            plt.imshow(
                chroma,
                aspect="auto",
                origin="lower",
                cmap="magma",
                interpolation="nearest",
            )
        elif isinstance(chroma, torch.Tensor):
            plt.imshow(
                chroma.numpy(),
                aspect="auto",
                origin="lower",
                cmap="magma",
                interpolation="nearest",
            )
        plt.colorbar(label="Velocity / Activation Strength")
        plt.title("Spec")
        plt.xlabel("Frame Index")
        plt.ylabel("Pitch Index")
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    preproc = MidrPreprocessor(config)

    # f_midi = "test_files/test_midi.MID"
    # print(preproc(f_midi))
    circle = Circle(config)
    pitches = np.array([0, 1, 11])
    velocities = np.array([64, 64, 64])
    circle.create_cloud(pitches, velocities)
    circle_cc = circle.center
    circle_ca = circle.area
    circle_cd = circle.diameter
    print(circle_cc)
    