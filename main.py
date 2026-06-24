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
        r (numpy.ndarray): Array containing 3D vector separations between molecules
        L (int): box width

    Returns:
        numpy.ndarray: separation with minimum image convention
    """
    r = np.where(r > L/2, r - L, np.where(r < L/2, r + L, r))
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
    horizontal = np.s_[np.newaxis, :, :]
    vertical = np.s_[:, np.newaxis, :]
    make_coordinate = np.s_[:, :, np.newaxis]

    # Max distance between particles for Leonard-Jones potential to apply
    rc = 3

    # Time derivatives of position values
    yt[pos] = y[vel]

    # Broadcast into 3x3x3 arrays to cover all molecule combinations
    r = np.subtract(y[pos][vertical], y[pos][horizontal])

    # Apply minimum image convention
    r = min_image(r, L)

    # Produce nxn array representing dot products of all particles separations with themselves
    r2 = np.sum(r**2, axis=2)
    r2[np.diag_indices_from(r2)] = 1

    # Produce nxn array representing forces acting on each molecule due to separations with other molecules
    f = 6*(2*r2**(-7) - r2**(-4))
    f[np.diag_indices_from(f)] = 0
    f = f[make_coordinate]*r

    # Sum forces along the rows to get net force on each particles
    yt[vel] = np.sum(f, axis=1)

    return yt

def verlet_integrate(y, delt):
    """
    Calculate new system state after a time step using velocity verlet method

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
    """
        Calculate new system state after a time step

        Args:
            y0 (numpy.ndarray): initial state matrix
            delt (float): time step

        Returns:
            numpy.ndarray: array representing system state at time delt
    """
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
