from core import *
import meshio
from meshreader import *
from matplotlib import pyplot as plt
import pygmsh

mesh_path = 'cur_square_plate.msh'
if True:
    # create geometry object
    geom = pygmsh.opencascade.Geometry(
        characteristic_length_min=0.0002,  # minimal element edge length
        characteristic_length_max=0.001,  # maximal element edges length
    )
    cut_circ = geom.add_disk([0.1, 0.1, 0], 0.08)
    drill_hole0 = geom.add_disk([0.09, 0.01, 0], 0.002)
    drill_hole1 = geom.add_disk([0.01, 0.09, 0], 0.002)
    square = geom.add_rectangle([0, 0, 0], 0.1, 0.1)  # horizontal rectangle
    geom.boolean_difference([square], [cut_circ, drill_hole0, drill_hole1])

    # generate and save the defined mesh
    mesh = pygmsh.generate_mesh(geom, msh_filename=mesh_path, dim=2)


# define fem
fem = FiniteElementMethod()

fem.open_mesh(mesh_path, material=copper)

# show all nodes on the first time to know on which nodes to apply dirichlet boundary conditions
# global_plot_all_nodes(show=True)

# configuring nodes
for node in fem.nodes:
    # dirichlet for electric potential
    if node.distance_to(Node(.09, .01)) <= 0.0021:
        fem.apply_dirichlet(node.id, 0.0)  # 1V to the left side of the bar
    if node.distance_to(Node(.01, .09)) <= 0.0021:
        fem.apply_dirichlet(node.id, 0.01)  # 0V to the right side of the bar



fem.solve_potential()
fem.plot_all_elements(numbering=False)
fem.plot_contour_potential(40, show=True)

fem.compute_E_field(show=True, vector=True)
fem.compute_powerlosses(depth=0.001)
fem.solve_temperature(hbot=0, htop=10, Tamb=300)
fem.plot_contour_temperature(show=True)
