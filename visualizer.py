import pyvista as pv
import numpy as np

pv.global_theme.allow_empty_mesh = True

def clip_box_closed(mesh, bounds):
    """
    Perform a closed surface clip on a mesh based on a bounding box

    Args:
        mesh (pyvista.PolyData): mesh to clip
        bounds (tuple): bounds to clip the mesh

    Returns:
        pyvista.PolyData: clipped mesh
    """
    mesh = mesh.clip_closed_surface('x', origin=[bounds[0], 0, 0])
    mesh = mesh.clip_closed_surface('-x', origin=[bounds[1], 0, 0])
    mesh = mesh.clip_closed_surface('y', origin=[0, bounds[2], 0])
    mesh = mesh.clip_closed_surface('-y', origin=[0, bounds[3], 0])
    mesh = mesh.clip_closed_surface('z', origin=[0, 0, bounds[4]])
    mesh = mesh.clip_closed_surface('-z', origin=[0, 0, bounds[5]])
    return mesh

def make_sphere(center, speed, bounds, radius):
    """
    Create a pyvista sphere with given center, speed and bounds

    Args:
        center (np.ndarray): center of the sphere
        speed (float): speed of the sphere
        bounds (tuple): bounds to clip the sphere
        radius (float): radius of particles

    Returns:
        float: relevant timestep
    """
    sphere = clip_box_closed(pv.Sphere(radius=radius, center=center), bounds)
    sphere["speed"] = np.full(sphere.n_points, speed)
    return sphere

def rel_timestep(Y, vel, radius):
    """
    Find the relevant timestep for visualization

    Args:
        Y (numpy.ndarray): Array of the system of particles
        vel (int): index of velocity values in system state matrix
        radius (float): radius of particles

    Returns:
        float: relevant timestep
    """
    speed_sum = 0
    n = len(Y[0][0])

    # Take halfway point as snapshot, allows system to settle into equilibrium
    half = len(Y)//2

    for i in range(n):
        speed_sum += np.linalg.norm(Y[half][vel][i])

    avg_speed = speed_sum/n
    timestep = radius/avg_speed
    return timestep

def visualize_particles(Y, delt, n, L, pos, vel, radius, output_file='md.gif'):
    """
    Visualize a system of particles using pyvista

    Args:
        Y (numpy.ndarray): (n+1)x4x2 array of the system of particles
        delt (float): time step
        n (int): number of particles
        output_file (string): Name of output GIF file
        pos (int): index of position values in system state matrix
        vel (int): index of velocity values in system state matrix
        radius (float): radius of particles

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
        speed_list.append(np.linalg.norm(Y[0][vel][i]))
        mol_list.append(0)
        clone_list.append(0)

    ratio = int(round(rel_timestep(Y, vel, radius)/delt))

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

            # Update speeds
            speed_list[j] = np.linalg.norm(Y[i*ratio][vel][j])

            # Add new meshes for molecules
            mol_list.append(pl.add_mesh(make_sphere(Y[i*ratio][pos][j], speed_list[j], bounds, radius), scalars="speed", clim=(0, 10)))

            # If the molecules leave the box, create clones
            if (Y[i*ratio][pos][j][0] + radius > coord_max or Y[i*ratio][pos][j][1] + radius > coord_max or
                Y[i*ratio][pos][j][2] + radius > coord_max or Y[i*ratio][pos][j][0] - radius < -coord_max or
                Y[i*ratio][pos][j][1] - radius < -coord_max or Y[i*ratio][pos][j][2] - radius < -coord_max):

                # Set displacement vectors which are the length of the box along the velocity vector
                if speed_list[j] != 0:
                    displacement = L * (Y[i*ratio][vel][j] / speed_list[j])
                else:
                    displacement = np.array([0.0, 0.0, 0.0])

                clone_pos = Y[i*ratio][pos][j] - displacement
                clone_list.append(pl.add_mesh(make_sphere(clone_pos, speed_list[j], bounds, radius), clim=(0, 10)))
            else:
                clone_list.append(0)

        # Create frame of animation
        pl.write_frame()

    pl.close()