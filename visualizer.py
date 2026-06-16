import pyvista as pv
import numpy as np

coord_max = 3

def visualize_particles(Y, output_file='md.gif'):
    """
    Visualize a system of particles using pyvista

    Args:
        Y (numpy.ndarray): (n+1)x4x2 array of the system of particles
        output_file (string): Name of output GIF file

    Returns:
        none
    """
    pl = pv.Plotter()
    sphere1 = pv.Sphere(radius=0.5)
    sphere2 = pv.Sphere(radius=0.5)
    box = pv.Box((-coord_max, coord_max, -coord_max, coord_max, -coord_max, coord_max))

    # Initial molecule speeds
    speed1 = np.linalg.norm(Y[0][2])
    speed2 = np.linalg.norm(Y[0][3])

    # Set speeds in sphere objects
    sphere1["speed1"] = np.full(sphere1.n_points, speed1)
    sphere2["speed2"] = np.full(sphere2.n_points, speed2)

    # Create "actors" for molecules and container
    mol1 = pl.add_mesh(sphere1, scalars='speed1', clim=(0, 10))
    mol2 = pl.add_mesh(sphere2, scalars='speed2', clim=(0, 10))
    container = pl.add_mesh(box, color='black', opacity=0.1, show_edges=True)
    container_edges = pl.add_mesh(box, style='wireframe', color='black', line_width=2)

    # Open gif for animation
    pl.open_gif(output_file)

    # Animation Loop
    for i in range(len(Y)):
        # Update molecule positions
        mol1.position = Y[i, 0]
        mol2.position = Y[i, 1]

        # Update speeds
        speed1 = np.linalg.norm(Y[i][2])
        speed2 = np.linalg.norm(Y[i][3])

        # Fill sphere objects with speed
        sphere1["speed1"][:] = speed1
        sphere2["speed2"][:] = speed2

        # Create frame of animation
        pl.write_frame()

    pl.close()