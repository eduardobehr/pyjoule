from core import *
from meshreader import *

# define geometry (geometry may also be imported with module meshreader)



# define fem
fem = FiniteElementMethod()

# option 1: manually load elements and nodes with MeshReader:
"""
mesh = MeshReader('meshes/drilled_plate.msh')
nodes = mesh.generate_nodes()
elements = mesh.generate_elements()
fem.set_elements(elements)
fem.set_nodes(nodes)
"""
# option 2: call open_mesh method to load mesh defined above:
fem.open_mesh('meshes/drilled_plate.msh')

fem.build_global_matrix()  

fem.apply_bc([0,10,11,12,13,14,15,16,17,18,19,20,21], 13*[12])  # positive
fem.apply_bc([1,22,23,24,25,26,27,28,29,30,31,32,33], 13*[0])  # negative



fem.solve()
fem.plot_all_elements(numbering=False)
fem.plot_contour_potential(15, show=False)
fem.plot_gradient(show=True)

print('Finished.')
