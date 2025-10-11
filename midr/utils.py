import numpy as np


def get_angular_mean(angles: np.array, T: float = 2*np.pi) -> float:
    """
    Finds the angular mean of a set of angles
    The first angle is always used as the reference as
    to where they should wrap around
    """
    angles = np.asarray(angles)
    ref_angle = angles[0]

    delta      = (angles - ref_angle + T/2) % T - T/2
    angle_mean = np.mean(ref_angle + delta) % T

    return angle_mean


def get_weighted_angular_mean(angles: np.array, weights: np.array, T: float = 2*np.pi) -> float:
    """
    Finds the weighted angular mean of a set of angles
    The first angle is always used as the reference as
    to where they should wrap around
    """
    angles = np.asarray(angles)
    ref_angle = angles[0]

    delta      = (angles - ref_angle + T/2) % T - T/2
    angle_mean = np.average(ref_angle + delta, weights) % T

    return angle_mean


def unwrap_angles(angles: np.array, T: float = 2*np.pi, ref_angle: float=0) -> float:
    angles = np.asarray(angles)
    return (ref_angle + (angles - ref_angle + T/2) % T - T/2) % T