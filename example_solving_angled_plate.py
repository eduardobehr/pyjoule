from core import *
from meshreader import *

# define geometry (geometry may also be imported with module meshreader)



# define fem
fem = FiniteElementMethod()

# option 1: manually load elements and nodes with MeshReader:
if False:
    mesh = MeshReader('meshes/drilled_plate.msh')
    nodes = mesh.generate_nodes()
    elements = mesh.generate_elements()
    fem.set_elements(elements)
    fem.set_nodes(nodes)
    fem._build_potential_global_matrix()
else:
    # option 2: call open_mesh method to load mesh defined above:
    fem.open_mesh('meshes/drilled_plate.msh', material=copper)



fem.apply_dirichlet([0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 13 * [1])  # positive
fem.apply_dirichlet([1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], 13 * [0])  # negative


fem.solve_potential()
fem.plot_all_elements(numbering=False)
fem.plot_contour_potential(15, show=True)
# fem.compute_E_field(_abs=False, show=True)
fem.compute_E_field(show=True, vector=True)
fem.compute_powerlosses(depth=0.001)
fem.solve_temperature(hbot=0, htop=100, Tamb=297)
fem.plot_contour_temperature(show=True)

print('Finished.')
