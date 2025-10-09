import numpy as np
import pickle
import matplotlib.pyplot as plt
import pretty_midi
import json
import librosa

from scipy.spatial.distance import pdist


class MidrPreprocessor:
    def __init__(self, config):
        self.fs = config['spec']['sr'] / config['spec']['hop_sample']

        self.circle = Circle(config)
        self.torus  = Torus(config)
        self.spiral = Spiral(config)

    def midi2chroma(self, f_midi):
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        assert len(midi.instruments) == 1, f"Model is finetuned for single instrument"
        
        chroma = midi.instruments[0].get_chroma(fs=self.fs, pedal_threshold=128)
        
        # arrange chroma in circle of fifths
        circle_of_fifths = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5])
        chroma = chroma[circle_of_fifths]
        # self.plot_chroma(chroma)
        
        return chroma.T
    
    def chroma2midr(self, chroma):
        nframe      = len(chroma)
        a_circle_cc = np.zeros((nframe, 2), dtype=np.float32)
        a_circle_ca = np.zeros((nframe, 1), dtype=np.float32)
        a_circle_cd = np.zeros((nframe, 1), dtype=np.float32)

        for i in range(nframe):
            # circle
            pitches = np.nonzero(chroma[i])[0]
            if pitches.size > 0:
                velocities = chroma[i][pitches]
                self.circle.create_cloud(pitches, velocities)
                a_circle_cc[i] = self.circle.center
                a_circle_ca[i] = self.circle.area
                a_circle_cd[i] = self.circle.diameter

        a_circle = {
            'circle_cc' : a_circle_cc.tolist(),
            'circle_ca' : a_circle_ca.tolist(),
            'circle_cd' : a_circle_cd.tolist()
        }
        a_midr = {
            'circle' : a_circle
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

class Torus:
    def __init__(self, config):
        self.r = config['midr']['torus']['radius_in']
        self.R = config['midr']['torus']['radius_out']
        self.weighted = config['midr']['torus']['weighted']

    def create_cloud(self, pitches, velocities):
        self.note_angles = self.get_note_angles(pitches)
        self.center = self.get_center(self.note_angles, velocities)
        self.area = self.get_area(self.note_angles)
        self.diameter = self.get_diameter(self.note_angles)

    def get_note_angles(self, pitches):
        note_angles = np.zeros((len(pitches), 2), dtype=np.float32)

        for i in range(len(pitches)):
            pitch_class = pitches[i] % 12
            phi     = 2 * np.pi * (pitch_class % 4) / 4
            theta   = 2 * np.pi * (pitch_class // 3) / 3
            note_angles[i] = [phi, theta]

        return note_angles
    
    def get_center(self, angles, velocities):
        phi     = angles[:,0]
        theta   = angles[:,1]

        if self.weighted:
            u_center = np.average(phi, weights=velocities)
            v_center = np.average(theta, weights=velocities)
        else:
            u_center = np.mean(phi)
            v_center = np.mean(theta)

        # Convert torus angles to 3D coordinates
        x = (self.R + self.r * np.cos(v_center)) * np.cos(u_center)
        y = (self.R + self.r * np.cos(v_center)) * np.sin(u_center)
        z = self.r * np.sin(v_center)

        return np.array([x, y, z])
    
    def get_area(self, angles):
        # Approximate polygon area on the torus by projecting to 2D in u-v space
        phi     = angles[:,0]
        theta   = angles[:,1]

        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        coords = np.column_stack((x, y))

        # Use shoelace formula in XY plane as approximation
        x_coords = coords[:,0]
        y_coords = coords[:,1]

        area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))
        return area
    
    def get_diameter(self, angles):
        phi     = angles[:,0]
        theta   = angles[:,1]

        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z = self.r * np.sin(theta)
        coords = np.column_stack((x, y, z))

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        diameter = np.max(dist_matrix)
        return diameter

    def plot(self, show_torus=True):
        u = self.note_angles[:,0]
        v = self.note_angles[:,1]

        x = (self.R + self.r * np.cos(v)) * np.cos(u)
        y = (self.R + self.r * np.cos(v)) * np.sin(u)
        z = self.r * np.sin(v)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot torus surface
        if show_torus:
            u_surf = np.linspace(0, 2*np.pi, 12)
            v_surf = np.linspace(0, 2*np.pi, 9)
            U, V = np.meshgrid(u_surf, v_surf)
            X = (self.R + self.r * np.cos(V)) * np.cos(U)
            Y = (self.R + self.r * np.cos(V)) * np.sin(U)
            Z = self.r * np.sin(V)
            ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3, rstride=1, cstride=1)

        # Plot notes
        ax.scatter(x, y, z, c='blue', label='Notes', s=100)
        # Plot center
        ax.scatter(self.center[0], self.center[1], self.center[2], c='red', label='Center', s=150)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Torus Note Cloud')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.show()


class Circle:
    def __init__(self, config):
        self.radius   = config['midr']['circle']['radius']
        self.weighted = config['midr']['circle']['weighted']
        
        self.note_angles = None
        self.center      = None
        self.area        = None
        self.diameter    = None

    def __repr__(self):
        return (
            f"Circle(radius={self.radius}, weighted={self.weighted}, "
            f"note_angles={self.note_angles}, center={self.center}, "
            f"area={self.area}, diameter={self.diameter})"
        )
    
    def create_cloud(self, pitches, velocities):
        self.note_angles = self.get_note_angles(pitches)

        self.center     = self.get_center(self.note_angles, velocities)
        self.area       = self.get_area(self.note_angles)
        self.diameter   = self.get_diameter(self.note_angles)

    def get_note_angles(self, pitches):
        return np.array(pitches, dtype=np.float32) * (np.pi / 6)

    def get_center(self, angles, velocities):
        # We get the CIRCULAR mean to account for loops
        x = np.cos(angles)
        y = np.sin(angles)

        # Get mean/weighted mean
        if self.weighted:
            x_mean = np.average(x, weights=velocities)
            y_mean = np.average(y, weights=velocities)
        else:
            x_mean = np.mean(x)
            y_mean = np.mean(y)

        center_angle = np.arctan2(y_mean, x_mean)
        
        # Convert to cartesian
        # TODO check if angular is better
        center = self.radius * np.array([np.sin(center_angle), np.cos(center_angle)])

        return center
    
    def get_area(self, angles):
        if len(angles) < 2:
            return 0.0
        
        y = self.radius * np.cos(angles)
        x = self.radius * np.sin(angles)

        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return area
    
    def get_diameter(self, angles):
        x = self.radius * np.sin(angles)
        y = self.radius * np.cos(angles)
        coords = np.column_stack((x, y))
        if len(coords) < 2:
            return 0.0
        return np.max(pdist(coords))

    def plot(self):
        y = self.radius * np.cos(self.note_angles)
        x = self.radius * np.sin(self.note_angles)

        plt.figure(figsize=(6,6))
        plt.scatter(x, y, c='blue', label='Notes', s=100)
        plt.scatter(self.center[0], self.center[1], c='red', label='Center', s=150)

        circle = plt.Circle((0, 0), self.radius, color='gray', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)

        coords = np.column_stack((x, y))
        coords = np.vstack((coords, coords[0]))
        plt.plot(coords[:,0], coords[:,1], c='green', linestyle='-', alpha=0.7, label='Note Polygon')

        plt.gca().set_aspect('equal', 'box')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Circle Note Cloud')
        plt.legend()
        plt.grid(True)
        plt.show()


class Spiral():
    def __init__(self, config):
        return
    

if __name__=="__main__":
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    preproc = MidrPreprocessor(config)

    f_midi = "test_files/test_midi.MID"
    print(preproc(f_midi))
    