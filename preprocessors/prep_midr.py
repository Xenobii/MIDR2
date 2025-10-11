import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import json


class MidrPreprocessor:
    def __init__(self, config):
        self.fs = config['spec']['sr'] / config['spec']['hop_sample']

        self.circle = Circle(config)
        self.spiral = Spiral(config)

    def midi2chroma(self, f_midi):
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        assert len(midi.instruments) == 1, f"Model is finetuned for single instrument"
        
        chroma = midi.instruments[0].get_chroma(fs=self.fs, pedal_threshold=128)
        
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

        for i in range(nframe):
            # circle
            pitches = np.nonzero(chroma[i])[0]
            if pitches.size > 0:
                velocities = chroma[i][pitches]
                self.circle.create_cloud(pitches, velocities)
                a_circle_cc[i] = self.circle.center
                a_circle_cd[i] = self.circle.diameter

        a_circle = {
            'circle_cc' : a_circle_cc.tolist(),
            'circle_cd' : a_circle_cd.tolist(),
        }
        a_spiral = {
            'spiral_cc' : a_spiral_cc.tolist(),
            'spiral_cd' : a_spiral_cd.tolist(),
        }
        a_midr = {
            'circle' : a_circle,
            'spiral' : a_spiral
        }
        return a_midr
    
    def __call__(self, x):
        x = self.midi2chroma(x)
        x = self.chroma2midr(x)
        return x
    
    def plot_chroma(self, chroma):
        chroma = np.array(chroma)
        if chroma.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {chroma.shape}")

        plt.figure(figsize=(10, 4))
        plt.imshow(
            chroma,
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
    