# import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.pyplot import semilogy
import visualizer as vs


def derivative(y):
    """
    Calculate time derivative of molecular state of system

    Args:
        y (numpy.ndarray): initial 4x2 state matrix

    Returns:
        numpy.ndarray: 4x2 matrix representing time derivative of system state
    """
    yt = np.zeros((4, 3))
    # Time derivatives of position values
    yt[0] = y[2]
    yt[1] = y[3]

    # Distance between 2 molecules
    r = np.subtract(y[0], y[1])

    # Time derivative of velocity values
    f = 6 * (2 * np.dot(r, r) ** (-7) - (np.dot(r, r)) ** (-4)) * r
    yt[2] = f
    yt[3] = -f

    return yt


def integrate(y0, delt):
    """
    Calculate new system state after a time step

    Args:
        y0 (numpy.ndarray): initial 4x2 state matrix
        delt (float): time step

    Returns:
        numpy.ndarray: 4x2 array representing system state at time delt
    """
    y1_pred = np.add(y0, delt * (derivative(y0)))
    y1 = np.add(y0, (delt / 2) * (np.add(derivative(y1_pred), derivative(y0))))
    return y1


def simulate(y0, tf, delt):
    """
    Simulate the system dynamics from t = 0 to t = tf using a fixed time step dt

    Args:
        y0 (numpy.ndarray): initial 4×2 state matrix
        tf (float): final simulation time
        dt (float): simulation time step

    Returns:
        numpy.ndarray: (n+1)x4x2 array where n = tf / dt,
        the i-th 4×2 matrix contains the system state at time t = i·dt
    """
    n = tf / delt
    Y = np.zeros((int(n + 1), 4, 3))
    y = y0

    for i in range(len(Y)):
        Y[i] = y
        y = integrate(y, delt)

    return Y


if __name__ == '__main__':
    r_min = 2 ** (1 / 6)

    x1 = np.array([0.5 * (r_min + 0.5), 0, 0])
    x2 = np.array([-0.5 * (r_min + 0.5), 0, 0])
    v1 = np.array([0, 0, 0])
    v2 = np.array([0, 0, 0])
    y0 = np.array([x1, x2, v1, v2])
    tf = 10
    delt = 0.01
    Y = simulate(y0, tf, delt)
    vs.visualize_particles(Y)

    """
    # (b)
    # Error(Log scale) vs Delta t graph
    x1 = np.array([0.5*(r_min + 0.1), 0])
    x2 = np.array([-0.5*(r_min + 0.1), 0])
    v1 = np.array([0, 0])
    v2 = np.array([0, 0])
    y0 = np.array([x1, x2, v1, v2])
    tf = 10
    delt = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
    yn_ref = simulate(y0, tf, 0.0005)[-1]
    err = np.zeros(len(delt) - 1)
    for i in range(len(delt) - 1):
        err[i] = np.linalg.norm(simulate(y0, tf, delt[i])[-1] - yn_ref, ord=2)
        delt = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
    print(np.polyfit(np.log10(delt), np.log10(err), 1))
    plt.loglog(delt, err, 'b-')
    plt.title('Error vs Delta t')
    plt.xlabel('Delta t')
    plt.ylabel('Error (log scale)')
    plt.show()

    # Separation Distance vs Time
    Y = simulate(y0, tf, 0.0005)
    x_diff = np.zeros(len(Y))
    t = np.arange(len(Y))*0.0005
    for i in range(len(Y)):
        x_diff[i] = np.linalg.norm(np.subtract(Y[i][0], Y[i][1]), ord=2)
    plt.plot(t, x_diff, 'r-')
    plt.title('Separation Distance vs Time')
    plt.xlabel('Time')
    plt.ylabel('Separation Distance')
    plt.show()
    """

    """ 
    # (c)
    tf = 225
    delt = 0.01
    x1 = np.array([-0.5*r_min, 0])
    x2 = np.array([0.5*r_min, 0])
    v1 = np.array([0.01, 0.01])
    v2 = np.array([-0.01, -0.01])
    y0 = np.array([x1, x2, v1, v2])
    Y = simulate(y0, tf, delt)
    for i in range(len(Y)):
        if i%(10**4) == 0:
            plt.quiver(Y[i][0][0], Y[i][0][1], Y[i][2][0], Y[i][2][1], color='r', scale=0.1, scale_units='xy')
            plt.quiver(Y[i][1][0], Y[i][1][1], Y[i][3][0], Y[i][3][1], color='b', scale=0.1, scale_units='xy')
    plt.show()

    b = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    proj1_mag = np.zeros(len(Y))
    comp1_mag = np.zeros(len(Y))
    proj2_mag = np.zeros(len(Y))
    comp2_mag = np.zeros(len(Y))
    for i in range(len(Y)):
        r = np.subtract(Y[i][0], Y[i][1])
        f = 6 * (2 * np.dot(r, r) ** (-7) - (np.dot(r, r)) ** (-4)) * r
        f1 = f
        f2 = -f

        # projection
        proj1 = np.dot(f1, b) * b
        proj2 = np.dot(f2, b) * b

        # orthogonal components
        comp1 = f1 - proj1
        comp2 = f2 - proj2

        proj1_mag[i] = np.linalg.norm(proj1)
        comp1_mag[i] = np.linalg.norm(comp1)
        proj2_mag[i] = np.linalg.norm(proj2)
        comp2_mag[i] = np.linalg.norm(comp2)

    # Plot
    t = np.arange(len(Y)) * delt
    plt.plot(t, proj1_mag, color='r')
    plt.plot(t, comp1_mag, color='b')
    plt.plot(t, proj2_mag, color='g')
    plt.plot(t, comp2_mag, color='y')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Force Projection and Orthogonal Component Magnitudes')
    plt.show()
    """
