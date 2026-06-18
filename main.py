import numpy as np
import visualizer as vs

L = 6

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

    # Time derivatives of position values
    yt[0] = y[1]
    yt[2] = y[3]

    # Distance between 2 molecules
    r = np.subtract(y[0], y[2])

    # Apply minimum image convention
    r = min_image(r, L)

    # Time derivative of velocity values
    f = 6 * (2 * np.dot(r, r) ** (-7) - (np.dot(r, r)) ** (-4)) * r
    yt[1] = f
    yt[3] = -f

    return yt


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


def simulate(y0, tf, delt):
    """
    Simulate the system dynamics from t = 0 to t = tf using a fixed time step delt

    Args:
        y0 (numpy.ndarray): initial state matrix
        tf (float): final simulation time
        dt (float): simulation time step

    Returns:
        numpy.ndarray: array of system states over time tf each separated by timestep delt
    """
    n = tf / delt
    Y = np.zeros((int(n + 1), 4, 3))
    y = y0

    for i in range(len(Y)):
        Y[i] = y
        y = integrate(y, delt)

        # wrap box if particles get out
        for k in range(3):
            y[0][k] = wrap_box(y[0][k], L)
            y[2][k] = wrap_box(y[2][k], L)

    return Y


if __name__ == '__main__':
    r_min = 2 ** (1 / 6)

    x1 = np.array([0.5 * (r_min + 0.5), 0, 0])
    v1 = np.array([2, 0, 0])
    x2 = np.array([-0.5 * (r_min + 0.5), 0, 0])
    v2 = np.array([0, 0, 0])
    y0 = np.array([x1, v1, x2, v2])
    tf = 10
    delt = 0.01
    Y = simulate(y0, tf, delt)
    vs.visualize_particles(Y, delt, 2, L)
