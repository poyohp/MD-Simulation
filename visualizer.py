import pyvista as pv
from networkx.classes.filters import show_edges
from sympy import true


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
    sphere = pv.Sphere(radius=0.5)
    box = pv.Box((-3, 3, -3, 3, -3, 3))

    # Create "actors" for molecules and container
    mol1 = pl.add_mesh(sphere, color='red')
    mol2 = pl.add_mesh(sphere, color='blue')
    container = pl.add_mesh(box, color='black', opacity=0.1, show_edges=True)
    container_edges = pl.add_mesh(box, style='wireframe', color='black', line_width=2)

    # Open gif for animation
    pl.open_gif(output_file)

    # Animation Loop
    for i in range(len(Y)):
        # Update molecule positions
        mol1.position = Y[i, 0]
        mol2.position = Y[i, 1]

        # Create frame of animation
        pl.write_frame()

    pl.close()