from meshreader import *
from matplotlib import pyplot as plt
import pygmsh

# create geometry object
geom = pygmsh.opencascade.Geometry(
  characteristic_length_min=0.001,  # mesh edges
  characteristic_length_max=0.01,  # mesh edges
  )

# BEGIN OF EDITABLE LINES --------------------------------------------------------------------

# shapes
vert_rect = geom.add_rectangle([0,0,0], 0.1, 0.5)  # vertical rectangle
horiz_rect = geom.add_rectangle([0,0,0], 0.5, 0.1)  # horizontal rectangle
vert_hole = geom.add_disk([0.05, 0.45, 0], 0.02)  # hole in the vertical rectangle
horiz_hole = geom.add_disk([0.45, 0.05, 0], 0.02)  # hole in the horizontal rectangle

# transformations
# join the rectangles
plate_union = geom.boolean_union([vert_rect, horiz_rect])

# subtract the holes from the rectangles
plate_drilled = geom.boolean_difference([plate_union], [vert_hole, horiz_hole])


# ALERT: change the name to avoid overwriting previous geometry!
mesh_path = 'meshes/nre_drilled_plate_2.msh'

# END OF EDITABLE LINES -----------------------------------------------------------------------------

# generate and save the defined mesh
mesh = pygmsh.generate_mesh(geom, msh_filename=mesh_path)#, geo_filename='bool_piece.geo', )

# load the mesh to verify it
loaded_mesh = MeshReader(mesh_path)
loaded_mesh.plot(grid=True, nodes=False)
nodes = loaded_mesh.generate_nodes()
elements = loaded_mesh.generate_elements()
#plot_all_nodes()
#plot_all_elements()
plt.show()
