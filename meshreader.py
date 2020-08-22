import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from typing import Tuple, List
from core import *



class MeshReader:
    """ Reads msh mesh file format """
    def __init__(self, path: str, *args):
        self.path = path
        self.mesh = meshio.read(path, *args)
        self.nodes = []
        self.elements = []
        self.vertices = self.mesh.points.transpose()[0:2]  # 2D, excludes z axis
        #self.triangles = self.mesh.cells[4][1]
        #print(self.mesh.cells_dict)
        if 'triangle' in self.mesh.cells_dict.keys():
            self.triangles = self.mesh.cells_dict['triangle']
        elif 'tetra' in self.mesh.cells_dict.keys():
            raise TypeError('No triangles found, but tetrahedrons instead. This means you are trying to open a 3D object, while this class only supports 2D.')
        else:
            raise TypeError('No triangles found.')
        self.trimesh = tri.Triangulation(*self.vertices, self.triangles)  # triangular mesh

    def plot(self, nodes=False, edges=True, grid=True):
        if nodes:
            plt.scatter(*self.vertices)
        if edges:
            plt.triplot(self.trimesh)
        if grid:
            plt.grid(True)
        plt.title(self.path)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
    def generate_nodes(self) -> List[Node]:
        """ Takes the x, y vertices and creates instances of Node """
        x, y = self.vertices
        assert len(x) == len(y), 'x and y must be of the same length'
        self.nodes = [Node(x[i], y[i]) for i in range(len(x))]
        return self.nodes

    def generate_elements(self, material=air) -> List[Element]:
        assert self.nodes, f'Nodes needed to generate elements. First call method {self.generate_nodes}.'
        for triang in self.triangles:
            nodes = []
            for node_id in triang:
                nodes.append(Node.get_node(node_id))
            self.elements.append(Element(nodes,material))
        return self.elements
        

if __name__ == '__main__':
    # TEST
    mesh=MeshReader('meshes/semicircle.msh')
    mesh.plot()
    nodes = mesh.generate_nodes()
    elements = mesh.generate_elements()
    plot_all_nodes()
    plot_all_elements()
    plt.show()
