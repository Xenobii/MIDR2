import numpy as np
from midr.utils import unwrap_angles
from scipy.spatial.distance import pdist



class Circle:
    def __init__(self, config):
        self.radius   = config['midr']['circle']['radius']
        self.weighted = config['midr']['circle']['weighted']
        
        self.note_angles = None
        self.center      = None
        self.area        = None
        self.diameter    = None

    def __repr__(self):
        return {
            f"Circle: {
                "radius     : ", self.radius,
                "weighted   : ", self.weighted,
                "center     : ", self.center,
                "diameter   : ", self.diameter,
                "area       : ", self.area
            }"
        }
    
    def get_note_angles(self, pitches):
        angles = np.array(pitches, dtype=np.float32) * (np.pi / 6)
        return unwrap_angles(angles, T=2*np.pi)

    def create_cloud(self, pitches, velocities):
        # Get note angles
        self.note_angles = self.get_note_angles(pitches)

        # Convert to cartessian
        x = self.radius * np.cos(self.note_angles)
        y = self.radius * np.sin(self.note_angles)

        # Get mean/weighted mean
        if self.weighted:
            x_mean = np.average(x, weights=velocities)
            y_mean = np.average(y, weights=velocities)
        else:
            x_mean = np.mean(x)
            y_mean = np.mean(y)

        self.center = np.array([x_mean, y_mean], dtype=np.float32)

        if len(pitches) > 1:
            self.diameter = np.max(pdist(np.stack([x, y], axis=1)))
        else:
            self.diameter = 0.0

    @classmethod
    def get_momentum(self, center, prev_center):
        return center - prev_center

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


class Spiral:
    def __init__(self, config):
        self.radius   = config['midr']['spiral']['radius']
        self.height   = config['midr']['spiral']['height']
        self.weighted = config['midr']['spiral']

        self.note_angles = None
        self.center      = None
        self.area        = None
        self.diameter    = None
    
    def __repr__(self):
        return {
            f"Spiral: {
                "radius     : ", self.radius,
                "height     : ", self.height,
                "weighted   : ", self.weighted,
                "center     : ", self.center,
                "diameter   : ", self.diameter,
                "area       : ", self.area
            }"
        }
    
    def get_note_angles(self, pitches):
        angles = np.array(pitches, dtype=np.float32) * (np.pi / 2)
        return unwrap_angles(angles, T=6*np.pi)
    
    def create_cloud(self, pitches, velocities):
        # Get note angles
        self.note_angles = self.get_note_angles(pitches)

        # Convert to cartessian
        x = self.radius * np.cos(self.note_angles)
        y = self.radius * np.sin(self.note_angles)
        z = self.height * self.note_angles / (12 * np.pi)

        # Get mean/weighted mean
        if self.weighted:
            x_mean = np.average(x, weights=velocities)
            y_mean = np.average(y, weights=velocities)
            z_mean = np.average(z, weights=velocities)
        else:
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            z_mean = np.mean(z)

        self.center = np.array([x_mean, y_mean, z_mean], dtype=np.float32)

        if len(pitches) > 1:
            self.diameter = np.max(pdist(np.stack([x, y, z], axis=1)))
        else:
            self.diameter = 0.0
    
    