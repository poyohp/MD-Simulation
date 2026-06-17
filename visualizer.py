import pyvista as pv
import numpy as np

radius = 0.5

def rel_timestep(Y, mol_num):
    d = 2*radius
    vel_sum = 0

    for i in range(len(Y)):
        vel_sum += np.sum(Y[i][2*mol_num - 1])

    avg_vel = vel_sum/len(Y)
    timestep = d/avg_vel
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
    pl = pv.Plotter()
    box = pv.Box((-coord_max, coord_max, -coord_max, coord_max, -coord_max, coord_max))
    pl.add_mesh(box, color='black', opacity=0.1, show_edges=True)
    pl.add_mesh(box, style='wireframe', color='black', line_width=2)
    sphere_list = []
    speed_list = []
    mol_list = []
    clone_list = []
    for i in range(n):
        sphere_list.append(pv.Sphere(radius=radius))

        # Initial molecule speeds
        speed_list.append(np.linalg.norm(Y[0, 2 * i + 1]))

        # Set speeds in sphere objects
        sphere_list[i]["speed"] = np.full(sphere_list[i].n_points, speed_list[i])

        # Create "actors" for molecules and container
        mol_list.append(pl.add_mesh(sphere_list[i], scalars="speed", clim=(0, 10)))
        clone_list.append(pl.add_mesh(sphere_list[i], scalars="speed", clim=(0, 10), opacity=0))

    # Open GIF for animation
    pl.open_gif(output_file)

    # Animation Loop
    for i in range(len(Y)):

        displacement_list = []

        for k in range(n):
            # Update molecule positions
            mol_list[k].position = Y[i, 2*k]

            # Update speeds
            speed_list[k] = np.linalg.norm(Y[i, 2*k + 1])

            # Set displacement vectors which are the length of the box along the velocity vector
            if speed_list[k] != 0:
                displacement_list.append(L * (Y[i, 2*k + 1] / speed_list[k]))
            else:
                displacement_list.append(np.array([0.0, 0.0, 0.0]))

            # Have clones trail molecules 1 box length away
            clone_list[k].position = mol_list[k].position - displacement_list[k]

            # If the molecules leave the box, visualize the clones
            if (mol_list[k].position[0] + radius > coord_max or mol_list[k].position[1] + radius > coord_max
                    or mol_list[k].position[2] + radius > coord_max or mol_list[k].position[0] < -coord_max
                    or mol_list[k].position[1] < -coord_max or mol_list[k].position[2] < -coord_max):
                clone_list[k].GetProperty().SetOpacity(1)
            else:
                clone_list[k].GetProperty().SetOpacity(0)

            # Fill sphere objects with speed
            sphere_list[k]["speed"][:] = speed_list[k]

        # Create frame of animation
        pl.write_frame()

    pl.close()