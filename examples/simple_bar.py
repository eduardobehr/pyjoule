from core import *
import meshio
from meshreader import *
from matplotlib import pyplot as plt
import pygmsh

mesh_path = 'simple_bar.msh'
if True:
    # create geometry object
    geom = pygmsh.opencascade.Geometry(
        characteristic_length_min=0.001,  # minimal element edge length
        characteristic_length_max=0.02,  # maximal element edges length
    )

    horiz_rect = geom.add_rectangle([0,0,0], 1, 0.1)  # horizontal rectangle
    # circ = geom.add_disk([0.5, 0.05, 0], 0.15)
    # geom.boolean_union([circ, horiz_rect])

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
    if node.x == 0:
        fem.apply_dirichlet(node.id, .1)  # 1V to the left side of the bar
    if node.x == 1:
        fem.apply_dirichlet(node.id, 0)  # 0V to the right side of the bar

for element in fem.elements:
    # changing conductivity
    # if element.get_centroid()[1] < 0:
    #     element.material = air
    # if element.get_centroid()[1] < 0.05:
    #     element.material = air
    ...


fem.solve_potential()
fem.plot_all_elements(numbering=False)
fem.plot_contour_potential(40, show=True)

fem.compute_E_field(show=True, vector=True)
fem.compute_powerlosses(depth=0.001)
fem.solve_temperature(hbot=0, htop=5, Tamb=300)
fem.plot_contour_temperature(show=True)
