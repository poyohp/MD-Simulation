import pyvista as pv
import numpy as np

radius = 0.5
pv.global_theme.allow_empty_mesh = True

def make_sphere(center, speed, bounds):
    """
    Create a pyvista sphere with given center, speed and bounds

    Args:
        center (np.ndarray): center of the sphere
        speed (float): speed of the sphere
        bounds (tuple): bounds to clip the sphere

    Returns:
        float: relevant timestep
    """
    sphere = pv.Sphere(radius=radius, center=center).clip_box(bounds, invert=False)
    sphere["speed"] = np.full(sphere.n_points, speed)
    return sphere

def rel_timestep(Y):
    """
    Find the relevant timestep for visualization

    Args:
        Y (numpy.ndarray): Array of the system of particles

    Returns:
        float: relevant timestep
    """
    d = 2*radius
    speed_sum = 0
    n = len(Y[0])/2

    # Take halfway point as snapshot, allows system to settle into equilibrium
    half = len(Y)//2

    for i in range(len(Y[half])):

        # Velocities are on odd indexes
        if i%2 != 0:
            speed_sum += np.linalg.norm(Y[half][i])

    avg_speed = speed_sum/n
    timestep = d/avg_speed
    return timestep

def visualize_particles(Y, delt, n, L, output_file='md.gif'):
    """
    Visualize a system of particles using pyvista

    Args:
        Y (numpy.ndarray): (n+1)x4x2 array of the system of particles
        delt (float): time step
        n (int): number of particles
        output_file (string): Name of output GIF file

    Returns:
        none
    """
    coord_max = L/2
    bounds = (-coord_max, coord_max, -coord_max, coord_max, -coord_max, coord_max)
    pl = pv.Plotter()
    box = pv.Box(bounds)
    pl.add_mesh(box, color='black', opacity=0.1, show_edges=True)
    pl.add_mesh(box, style='wireframe', color='black', line_width=2)
    speed_list = []
    mol_list = []
    clone_list = []

    for i in range(n):
        # Initial molecule speeds
        speed_list.append(np.linalg.norm(Y[0, 2 * i + 1]))
        mol_list.append(0)
        clone_list.append(0)

    ratio = int(round(rel_timestep(Y)/delt))

    # Open GIF for animation
    pl.open_gif(output_file)

    # Animation Loop
    for i in range(len(Y)//ratio):

        # Remove actors of previous timestep
        for j in range(n):
            if mol_list[j] != 0:
                pl.remove_actor(mol_list[j])
            if clone_list[j] != 0:
                pl.remove_actor(clone_list[j])

        mol_list.clear()
        clone_list.clear()

        for j in range(n):
            pos = Y[i*ratio, 2*j]
            vel = Y[i*ratio, 2*j + 1]

            # Update speeds
            speed_list[j] = np.linalg.norm(vel)

            # Add new meshes for molecules
            mol_list.append(pl.add_mesh(make_sphere(pos, speed_list[j], bounds), scalars="speed", clim=(0, 10)))

            # If the molecules leave the box, create clones
            if (pos[0] + radius > coord_max or pos[1] + radius > coord_max or
                pos[2] + radius > coord_max or pos[0] - radius < -coord_max or
                pos[1] - radius < -coord_max or pos[2] - radius < -coord_max):

                # Set displacement vectors which are the length of the box along the velocity vector
                if speed_list[j] != 0:
                    displacement = L * (vel / speed_list[j])
                else:
                    displacement = np.array([0.0, 0.0, 0.0])

                clone_pos = pos - displacement
                clone_list.append(pl.add_mesh(make_sphere(clone_pos, speed_list[j], bounds), clim=(0, 10)))
            else:
                clone_list.append(0)

        # Create frame of animation
        pl.write_frame()

    pl.close()