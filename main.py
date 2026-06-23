import numpy as np
import random
import visualizer as vs

# Box length
L = 10

# Number of particles
n = 10

# Mass of each particle
m = 10

# Radius of particles
radius = 0.5

# Index of position values in state matrix
pos = 0

# Index of velocity values in state matrix
vel = 1

def min_image(r, L):
    """
    Apply the minimum image convention

    Args:
        r (numpy.ndarray): 3D vector seperation between molecules
        L (int): box width

    Returns:
        numpy.ndarray: separation with minimum image convention
    """
    for i in range(len(r)):
        if r[i] > L / 2:
            r[i] -= L
        elif r[i] < -L / 2:
            r[i] += L
    return r

def wrap_box(x, L):
    """
    Wrap the molecule to the other side of the box

    Args:
        x (numpy.ndarray): 3D vector molecule position
        L (int): box width

    Returns:
        numpy.ndarray: 3D vector molecule position after translation
    """
    if x > L/2:
        x -= L
    elif x < -L/2:
        x += L
    return x

def derivative(y):
    """
    Calculate time derivative of molecular state of system

    Args:
        y (numpy.ndarray): initial state matrix

    Returns:
        numpy.ndarray: matrix representing time derivative of system state
    """
    yt = np.zeros(y.shape)

    # Max distance between particles for Leonard-Jones potential to apply
    rc = 3

    # Time derivatives of position values
    yt[pos] = y[vel]

    for i in range(n):
        for j in range(i + 1, n):

            # Distance between 2 molecules
            r = np.subtract(y[pos][i], y[pos][j])

            # Apply minimum image convention
            r = min_image(r, L)

            # Time derivative of velocity values
            if np.linalg.norm(r) <= rc:
                f = 6 * (2 * np.dot(r, r) ** (-7) - (np.dot(r, r)) ** (-4)) * r
                yt[vel][i] += f
                yt[vel][j] -= f

    return yt


def verlet_integrate(y, delt):
    """
    Calculate new system state after a time step

    Args:
        y (numpy.ndarray): initial state matrix
        delt (float): time step

    Returns:
        numpy.ndarray: array representing system state at time delt
    """
    # Get forces for each particle
    f = derivative(y)[1]
    delt2 = delt/2
    K = 0

    # First half update
    y[vel] += f*delt2/m
    y[pos] += y[vel] * delt

    # Update force with first half update
    f = derivative(y0)[1]

    # Second half update
    y[vel] += f*delt2/m
    K += m * np.sum(y[vel] * y[vel] / 2)

    return y


def integrate(y0, delt):
    y1_pred = np.add(y0, delt * (derivative(y0)))
    y1 = np.add(y0, (delt / 2) * (np.add(derivative(y1_pred), derivative(y0))))
    return y1


def simulate(y0, tf, delt, n):
    """
    Simulate the system dynamics from t = 0 to t = tf using a fixed time step delt

    Args:
        y0 (numpy.ndarray): initial state matrix
        tf (float): final simulation time
        dt (float): simulation time step

    Returns:
        numpy.ndarray: array of system states over time tf each separated by timestep delt
    """
    timesteps = tf / delt
    Y = np.zeros((int(timesteps + 1), 2, n, 3))
    y = y0

    for i in range(len(Y)):
        Y[i] = y
        y = verlet_integrate(y, delt)

        # wrap box if particles get out
        for j in range(n):
            for k in range(3):
                y[pos][j][k] = wrap_box(y[pos][j][k], L)

    return Y


if __name__ == '__main__':
    r_min = 2 ** (1 / 6)
    coord_max = L/2
    vel_max = 2
    y0 = np.zeros((2, n, 3))

    y0[pos] = np.random.uniform(-coord_max, coord_max, size=(n, 3))
    y0[vel] = np.random.uniform(-vel_max, vel_max, size=(n, 3))

    tf = 10
    delt = 0.01
    Y = simulate(y0, tf, delt, n)
    vs.visualize_particles(Y, delt, n, L, pos, vel, radius)
