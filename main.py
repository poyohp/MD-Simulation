import numpy as np
import random
import visualizer as vs

# Box length
L = 10

# Number of particles
n = 10

# Mass of each particle
m = 1

# Radius of particles
radius = 0.5

# Index of position values in state matrix
pos = 0

# Index of velocity values in state matrix
vel = 1

# Boltzmann constant (reduced LJ units)
kb = 1

# Max distance between particles for Leonard-Jones potential to apply
rc = 3

# Temperature of system
temp = 1.0

random.seed()

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

def compute_temperature(vel):
    """
    Compute temperature of system

    Args:
        vel (numpy.ndarray): Array containing 3D vector particle velocities

    Returns:
        float: temperature of system
    """
    N = len(vel)
    KE = 0.5 * np.sum(vel**2)

    # Derived from equipartition formula
    T = (2 * KE) / (3 * N * kb)

    return T

def lowe_ander_thermo(r, vel, delt):
    """
    Apply Lowe-Anderson thermostat to system

    Args:
        r (numpy.ndarray): Array containing 3D vector separations between particles
        vel (numpy.ndarray): Array containing 3D vector particle velocities
        delt (float): time step

    Returns:
        numpy.ndarray: Array containing 3D vector particle velocities with thermostat applied
    """
    nu = 0.1
    dist = np.linalg.norm(r, axis=2)

    # i and j are nd arrays that together store indexes that contain values between 0 and rc
    # i represents indexes of first particles, j represents indexes of second
    (i, j) = np.where((dist < rc) & (dist > 0))

    # Only consider upper triangular portion of dist to remove duplicates
    upper_triangular = i < j
    i = i[upper_triangular]
    j = j[upper_triangular]

    # Take values in r and dist that fit requirements, note that filtered arrays have 1 fewer dimension than the originals
    r_filtered = r[i, j]
    dist_filtered = dist[i, j]

    # Obtain array of normalized separation vectors
    normalized_r = r_filtered/dist_filtered[:, np.newaxis]

    # Get difference in velocity for particle pairs that satisfy 0 < separation < rc
    v_diff = vel[j] - vel[i]

    # Elementwise multiplication and addition along the rows acts as row-wise dot products
    v_rel_old = np.sum(v_diff * normalized_r, axis=1)

    # Draw new velocities from Maxwell-Boltzmann distribution
    v_rel_new = np.random.normal(0, np.sqrt(temp), size=len(i))

    # Lowe-Andersen collision condition
    collide = np.random.rand(len(i)) < (nu * delt)

    # Only apply changes on colliding particles
    delta_v_scalar = np.zeros(v_rel_old.shape)
    delta_v_scalar[collide] = v_rel_new[collide] - v_rel_old[collide]
    delta_v = delta_v_scalar[:, np.newaxis] * normalized_r

    # Update velocities
    vel[i] -= 0.5 * delta_v
    vel[j] += 0.5 * delta_v

    return vel


def derivative(y, return_r=False):
    """
    Calculate time derivative of molecular state of system

    Args:
        y (numpy.ndarray): initial state matrix
        return_r (bool): option to skip taking the derivative and return particle separations

    Returns:
        numpy.ndarray: matrix representing time derivative of system state
    """
    yt = np.zeros(y.shape)
    horizontal = np.s_[np.newaxis, :, :]
    vertical = np.s_[:, np.newaxis, :]
    make_coordinate = np.s_[:, :, np.newaxis]

    # Time derivatives of position values
    yt[pos] = y[vel]

    # Broadcast into 3x3x3 arrays to cover all molecule combinations
    r = np.subtract(y[pos][vertical], y[pos][horizontal])

    # Apply minimum image convention
    r = min_image(r, L)

    r = np.where(r >= rc, rc + 1, r)

    if return_r:
        return r

    # Produce nxn array representing dot products of all particles separations with themselves
    r2 = np.sum(r**2, axis=2)
    r2[np.diag_indices_from(r2)] = rc + 1

    # Produce nxn array representing forces acting on each molecule due to separations with other molecules
    f = np.where(r2 != rc + 1, 6*(2*r2**(-7) - r2**(-4)), r2)
    f = np.where(f == rc + 1, 0, f)

    # Extend nxn array to nxnx3 to represent vectors
    f = f[make_coordinate]*r

    # Sum forces along the rows to get net force on each particle
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
    f = derivative(y)[1]

    # Second half update
    y[vel] += f*delt2/m
    K += m * np.sum(y[vel] * y[vel] / 2)

    y[vel] = lowe_ander_thermo(derivative(y, True), y[vel], delt)

    print("Temperature:", compute_temperature(y[vel]))

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
