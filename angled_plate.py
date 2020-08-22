from core import *
from meshreader import *

# define geometry (geometry may also be imported with module meshreader)
mesh=MeshReader('meshes/drilled_plate.msh')
nodes = mesh.generate_nodes()
elements = mesh.generate_elements()
plot_all_nodes()
mesh.plot()


# define fem
fem = FiniteElementsMethod(elements=Element.instances, nodes=Node.instances)
fem.build_global_matrix()  

fem.apply_bc([0,10,11,12,13,14,15,16,17,18,19,20,21], 13*[12])  # positive
fem.apply_bc([1,22,23,24,25,26,27,28,29,30,31,32,33], 13*[0])  # negative



fem.solve()
plot_all_elements(numbering=False)
#track_mouse()
plot_contour_potential(21, show=False)
fem.plot_gradient(show=True)

print('Finished.')
