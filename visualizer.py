import pyvista as pv
import numpy as np

coord_max = 3
L = 2*coord_max
radius = 0.5

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
    box = pv.Box((-coord_max, coord_max, -coord_max, coord_max, -coord_max, coord_max))
    sphere1 = pv.Sphere(radius=radius).clip_box(box, invert=False)
    sphere2 = pv.Sphere(radius=radius).clip_box(box, invert=False)

    # Initial molecule speeds
    speed1 = np.linalg.norm(Y[0][1])
    speed2 = np.linalg.norm(Y[0][3])

    # Set speeds in sphere objects
    sphere1["speed1"] = np.full(sphere1.n_points, speed1)
    sphere2["speed2"] = np.full(sphere2.n_points, speed2)

    # Create "actors" for molecules and container
    mol1 = pl.add_mesh(sphere1, scalars='speed1', clim=(0, 10))
    mol1_clone = pl.add_mesh(sphere1, scalars='speed1', clim=(0, 10), opacity=0)
    mol2 = pl.add_mesh(sphere2, scalars='speed2', clim=(0, 10))
    mol2_clone = pl.add_mesh(sphere2, scalars='speed2', clim=(0, 10), opacity=0)
    pl.add_mesh(box, color='black', opacity=0.1, show_edges=True)
    pl.add_mesh(box, style='wireframe', color='black', line_width=2)

    # Open gif for animation
    pl.open_gif(output_file)

    # Animation Loop
    for i in range(len(Y)):
        # Update molecule positions
        mol1.position = Y[i, 0]
        mol2.position = Y[i, 2]

        # Update speeds
        speed1 = np.linalg.norm(Y[i, 1])
        speed2 = np.linalg.norm(Y[i, 3])

        """
        sphere1 = pv.Sphere(radius=radius, center=Y[i, 0]).clip_box(box, invert=False)
        sphere2 = pv.Sphere(radius=radius, center=Y[i, 2]).clip_box(box, invert=False)
        sphere1["speed1"] = np.full(sphere1.n_points, speed1)
        sphere2["speed2"] = np.full(sphere2.n_points, speed2)

        mol1.mapper.dataset.copy_from(sphere1)
        mol2.mapper.dataset.copy_from(sphere2)
        mol1_clone.mapper.dataset.copy_from(sphere1)
        mol2_clone.mapper.dataset.copy_from(sphere2)
        """

        # Set displacement vectors which are the length of the box along the velocity vector
        displacement1 = L*(Y[i, 1]/speed1)
        displacement2 = L*(Y[i, 3]/speed2)

        # Have clones trail molecules 1 box length away
        mol1_clone.position = mol1.position - displacement1
        mol2_clone.position = mol2.position - displacement2

        # If the molecules leave the box, visualize the clones
        if (mol1.position[0] + radius > coord_max or mol1.position[1] + radius > coord_max
                or mol1.position[2] + radius > coord_max or mol1.position[0] < -coord_max
                or mol1.position[1] < -coord_max or mol1.position[2] < -coord_max):
            mol1_clone.GetProperty().SetOpacity(1)
        else:
            mol1_clone.GetProperty().SetOpacity(0)

        if (mol2.position[0] + radius > coord_max or mol2.position[1] + radius > coord_max
                or mol2.position[2] + radius > coord_max or mol2.position[0] < -coord_max
                or mol2.position[1] < -coord_max or mol2.position[2] < -coord_max):
            mol2_clone.GetProperty().SetOpacity(1)
        else:
            mol2_clone.GetProperty().SetOpacity(0)

        # Fill sphere objects with speed
        sphere1["speed1"][:] = speed1
        sphere2["speed2"][:] = speed2

        # Create frame of animation
        pl.write_frame()

    pl.close()