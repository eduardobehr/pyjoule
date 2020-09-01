# Author: Eduardo Eller Behr
#
#
#
# Reference: Ida, Nathan. 'Engineering Electromagnetics', 3rd Ed. Springer Verlag
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import tri
from numpy import sqrt, mean, ndarray
from typing import Tuple, List, Union


def global_plot_all_nodes(markersize=25, show=False):
    """ Plots all instances of class Node defined in the global scope """
    if not Node.instances:
        print('Warning: no instance of Node found')
        return
    for node in Node.instances:
        plt.plot(*node.get_location(), '.', markersize=markersize, c='k')
        x, y = node.get_location()

        plt.text(x, y, s=node.id, c='w', ha='center', va='center')
    if show:
        plt.show()


class Material:
    instances = []
    def __init__(self, name: str, electrical_conductivity: float, thermal_conductivity: float, permittivity: float = 1,
                 rgb: Tuple[float, float, float] = (1, 1, 1)):
        """
        :param name: str.
        :param electrical_conductivity: float.
        :param thermal_conductivity: float.
        :param permittivity: float.
        :param rgb: Tuple[float, float, float]. Tuple with 3 rgb values ranging from 0.0 to 1.0.
        """
        self.name = name
        self.cond_elect = electrical_conductivity  # [S/m]
        self.cond_therm = thermal_conductivity  # [W/(mK)]
        self.permittivity = permittivity  # [F/m]
        if len(rgb) != 3:
            raise ValueError('Parameter rgb must be a tuple of length 3.')
        self.color = rgb  # (0:1, 0:1, 0:1)
        self.instances.append(self)

    def __repr__(self):
        return f"""{self.name}:
        Electrical Conductivity: {self.cond_elect} [Siemens]
        Thermal Conductivity: {self.cond_therm} [UNIT]"""
    pass


copper = Material('Copper', 5.96e7, 385, 1, rgb=(1., 128 / 255, 0.))
air = Material('Air', 1e-9, 0.024, 1, rgb=(206 / 255, 1., 1.))
iron = Material('Iron', 1e7, 79.5, 1, rgb=(.5, .5, .5))
gold = Material('Gold', 4.11e7, 314, 1, rgb=(.9, .9, 0.))


class Node:
    instances = []  # to keep track of all instances
    count = 0  # counts the instances
    id_dict = {}  # {obj:obj.id for obj in instances}
    
    @classmethod
    def clear_instances(cls):
        """ clears instances """
        for i in range(len(cls.instances)-2):
            del cls.instances[i]  # FIXME
        cls.count = 0

    @classmethod
    def inc(cls):
        """ increments class variable count """
        cls.count += 1

    @classmethod
    def update_id(cls, dictionary):
        cls.id_dict.update(dictionary)

    @classmethod
    def get_node(cls, instance_id):
        """ Returns Node object whose id is given """
        return cls.id_dict[instance_id]

    def __init__(self, x: Union[int, float], y: Union[int, float],
                 init_temp: Union[int, float] = 0, init_potential: Union[int, float] = 0,
                 lock_temp: bool = False, lock_potent: bool = False):
        self.x = x
        self.y = y
        self.temperature = init_temp
        self.potential = init_potential
        self.temperature_locked = lock_temp
        self.potential_locked = lock_potent

        self.in_elements = []  # stores the elements of which this node is part of
        self.instances.append(self)
        self.id = self.count
        self.update_id({self.id: self})

        self.inc()

        pass

    def assign_to_element(self, element: 'Element'):
        if not isinstance(element, Element):
            raise TypeError(f'{element} is not {Element}')
        self.in_elements.append(element)
        
    def is_connected(self):
        """ Detect if a node is not part of any element to help avoid singular matrices """
        # ALERT TODO: find a way to exclude unconnected nodes from global matrix without messing up the index numbering!
        if self.in_elements == []:
            return False
        else:
            return True

    def __repr__(self):
        return f"\nNode {self.id} at {self.get_location()}. " \
               f"Temp: {self.get_temperature()} K. Potential: {self.potential}V"

    def get_location(self):
        return self.x, self.y

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def distance_to(self, other_node: 'Node'):
        return sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)

    def lock_temperature(self, arg: bool):
        self.temperature_locked = arg

    def lock_potential(self, arg: bool):
        self.potential_locked = arg

    def is_node(self, obj, type_error=True):
        """ Checks whether obj is an instanceof this class """
        if isinstance(obj, self.__class__):
            return True
        else:
            if type_error:
                raise TypeError(f'{obj} is no instance of {self.__class__}')
            return False

    def average_from_elements(self, element_attribute: str):
        """
         Returns the average value of given attributes of neighboring elements
        :param element_attribute: str. Name of the attribute of Element instance.
        :return: float
        """
        if not hasattr(Element.instances[0], element_attribute):
            raise ValueError(f'Element has no attribute {element_attribute}')
        values = []
        for element in self.in_elements:
            attr = element.__getattribute__(element_attribute)
            if attr is None:
                print(f'Warning: Element {element} has attribute {element_attribute} set to {None}.')

            values.append(attr)
        # if not values:
        if not self.is_connected():
            # print(f'Warning: Node {self.id} at {self.get_location()} is not part of any element. Returning 0.')
            return 0
        else:
            return float(np.mean(values))
    pass


class Element:
    instances = []  # to keep track of all instances
    id_dict = {}  # {obj:obj.id for obj in instances}
    count = 0  # counts the instances
    
    @classmethod
    def clear_instances(cls):
        """ clears instances """
        for i in range(len(cls.instances)-2):
            del cls.instances[i]  # FIXME
        cls.count = 0

    @classmethod
    def inc(cls):
        """ increments class variable count """
        cls.count += 1

    @classmethod
    def get_element(cls, instance_id):
        """ Returns Element object whose id is given """
        # ref = {obj:obj.id for obj in instances}
        return cls.id_dict[instance_id]

    def __init__(self, nodes: Tuple[Node, Node, Node], material: Material, charge_density: float = 0):
        """
        Creates a triangular element
        :param nodes: Tuple['Node']
        :param material: Material
        """
        if not isinstance(nodes, (tuple, list, ndarray)):
            raise TypeError('Parameter nodes must be a tuple of 3 nodes')
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(f'Object {node} is not instance of {Node}')
        self.nodes = tuple(nodes)
        if len(self.nodes) != 3:
            raise ValueError("An element must be triangular (exactly 3 nodes)")
        self.charge_density = charge_density  # volumetric charge density
        self.area = self.get_area()  # area of the element (triangular)
        self.power_loss = 0
        self.temperature = None
        self.assign_nodes_to_element()
        self.material = material
        self.instances.append(self)
        self.id = self.count
        self.id_dict.update({self.id: self.instances})
        self.inc()
        self.E_abs = None  # absolute value of electric field
        self.neighbors = None
        
        pass

    def __repr__(self) -> str:
        return f"Element {self.id} with nodes {[node.id for node in self.nodes]} and area {np.round(self.area, 6)}.\n"

    def set_charge_density(self, value: float):
        self.charge_density = float(value)

    def assign_nodes_to_element(self) -> None:
        # Note: force nodes in a counterclockwise indexation in the element
        # at least garantee they are ordered in the same rotational direction
        # at first, meshio and gmsh got all ordered in the clockwise direction
        # it they always do so and it works, let it be.
        # Else, swap the directions of the list with self.nodes = self.nodes[::-1]
        for _node in self.nodes:
            if not isinstance(_node, Node):
                raise TypeError(f'{_node} is not instance of {Node}')
            _node.assign_to_element(self)

    def get_vertices(self) -> Tuple[List[int], List[int]]:
        """ Returns the coordinates of all nodes, with the first replicated """
        x = [node.get_location()[0] for node in self.nodes]
        x.append(x[0])
        y = [node.get_location()[1] for node in self.nodes]
        y.append(y[0])
        return x, y

    def get_triangle(self) -> List[int]:
        """ Returns a list of 3 nodes id to reference the triangle vertices """
        return [self.nodes[i].id for i in range(3)]

    def get_centroid(self) -> Tuple[float, float]:
        """ Returns the (x,y) coordinates of the centroid of a triangle (mean of each axis' coordinates points) """
        x_coord = [node.get_location()[0] for node in self.nodes]
        y_coord = [node.get_location()[1] for node in self.nodes]
        x_cent = float(mean(x_coord))
        y_cent = float(mean(y_coord))
        return x_cent, y_cent

    def get_area(self) -> float:
        """ Returns the area of the triangle computed with the vertices coordinates """
        x0, x1, x2 = [self.nodes[i].x for i in range(3)]
        y0, y1, y2 = [self.nodes[i].y for i in range(3)]
        return abs((x0 - x2) * (y1 - y0) - (x0 - x1) * (y2 - y0)) / 2

    def get_elemental_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the elemental matrix
        Reference:
        Ida, Nathan. 'Engineering Electromagnetics', 3rd Ed. Springer Verlag
        Page 319
        s: 3x3 symmetric matrix
        q: 3x1 matrix
        """
        x, y = self.get_vertices()  # ignore last index, since it's equal to the first
        i, j, k = 0, 1, 2

        # Are these even needed?
        # pi = x[j]*y[k] - x[k]*y[j]
        # pj = x[k]*y[i] - x[i]*y[k]
        # pk = x[i]*y[j] - x[j]*y[i]

        qi = y[j] - y[k]
        qj = y[k] - y[i]
        qk = y[i] - y[j]

        ri = x[k] - x[j]
        rj = x[i] - x[k]
        rk = x[j] - x[i]

        s11 = qi * qi + ri * ri
        s12 = qi * qj + ri * rj
        s13 = qi * qk + ri * rk
        s21 = s12
        s22 = qj * qj + rj * rj
        s23 = qj * qk + rj * rk
        s31 = s13
        s32 = s23
        s33 = qk * qk + rk * rk

        s = np.array([
            [s11, s12, s13],
            [s21, s22, s23],
            [s31, s32, s33]
        ])

        q = self.charge_density * self.area / 3 * np.array([[1], [1], [1]])
        return s, q

    def local2global(self, local_index: int) -> int:
        """ Converts local index to global index
        :param local_index: int. Number 0, 1 or 2 (i, j, k) used to refer to the vertices of the element
        :return global_index: int. Number 0:n used to refer to all the vertices of the whole domain,
        where n-1 is the total number of vertices (nodes).
        """
        if local_index in [0, 1, 2]:
            global_index = self.nodes[local_index].id
            return global_index
        else:
            raise ValueError('Parameter local_index must be 0, 1 or 2')

    def global2local(self, global_index: int) -> int:
        """ Converts global index to local index
        :param global_index: int. Number 0:n used to refer to all the vertices of the whole domain,
        where n-1 is the total number of vertices (nodes).
        :return local_index: int. Number 0, 1 or 2 that refers to the local vertex of the element
        """
        _node = Node.instances[global_index]
        if _node in self.nodes:
            local_index = self.nodes.index(_node)
            return local_index
        else:
            raise ValueError(f'Node {_node.id} is not part of element {self.id}')

    def get_E_field(self):
        """ Returns the negative of the gradient of the potential at the element"""
        x, y = self.get_vertices()  # ignore last index, since it's equal to the first
        i, j, k = 0, 1, 2
        V = [None, None, None]
        V[i], V[j], V[k] = [self.nodes[l].potential for l in range(3)]

        q = [None, None, None]
        # for dV/dx
        q[i] = y[j] - y[k]
        q[j] = y[k] - y[i]
        q[k] = y[i] - y[j]

        r = [None, None, None]
        # for dV/dy
        r[i] = x[k] - x[j]
        r[j] = x[i] - x[k]
        r[k] = x[j] - x[i]

        partial_x = sum([q[l] * V[l] for l in [i, j, k]]) / (2 * self.area)
        partial_y = sum([r[l] * V[l] for l in [i, j, k]]) / (2 * self.area)

        return -partial_x, -partial_y

    def get_neighbors(self):
        # TODO
        if self.neighbors is None:
            node0, node1, node2 = self.nodes
            edges = ((node0, node1),
                     (node1, node2),
                     (node2, node0)
            )
            neighbors = []
            for edge in edges:
                ep0, ep1 = set(edge[0].in_elements), set(edge[1].in_elements)  # elements of point pair
                intersec = ep0 & ep1  # gets the triangles that have these two vertices
                if self not in intersec:  # self must be in intersect because both nodes belong to self
                    raise Exception(f'Element {self.id} is not in {intersec} whilst it should be by definition')

                intersec.remove(self)  # because self is not a neighbor of itself
                if len(intersec) > 1:
                    raise Exception(f'Nodes {edge[0]} and {edge[1]} have more than 2 elements in common.')

                if len(intersec) == 1:
                    neighbors.append(list(intersec)[0])
                else:
                    continue

            if len(neighbors) > 3:
                raise Exception(f'Element {self.id} has {len(neighbors)} neighbors'
                                f' and this should not be possible for a triangle')
            self.neighbors = neighbors
        return self.neighbors

def global_plot_all_elements(show=False, numbering=True, fill=False):
    """ Plots all instances of class Element defined in the global scope """
    if not Element.instances:
        print('Warning: no instance of Element found')
        return
    for element in Element.instances:
        plt.plot(*element.get_vertices(), c='black', linewidth=.1)
        # print(f'Element {element.id}: {element.get_vertices()}')
        if fill:
            plt.fill(*element.get_vertices(), color=element.material.color)
            ...
        if numbering:
            x, y = element.get_centroid()
            plt.text(x, y, s=str(element.id), c='k', ha='center', va='center')
    if show:
        plt.show()


def global_plot_contour_potential(levels: int = 21, show: bool = False, cmap: str = 'jet'):
    """
    Plots the contour of the electric potential scalar field for nodes and elements defined in the global scope
    :param levels: int. Number of contour levels
    :param show: bool. Whether or not to show the plot (calls pyplot.show if True)
    :param cmap: str. String representing the cmap argument (colormap) of pyplot.tricontour
    :return: None
    """
    if levels < 2:
        raise ValueError('Contour plot needs at least 2 levels to be correctly displayed')
    elements = Element.instances
    triangles = [element.get_triangle() for element in elements]
    nodes = Node.instances
    x = np.array([node.get_location()[0] for node in nodes])
    y = np.array([node.get_location()[1] for node in nodes])
    triangulation = tri.Triangulation(x, y, triangles)
    potentials = [node.potential for node in nodes]
    plt.tricontourf(triangulation, potentials, levels=levels-1, cmap=cmap)
    plt.colorbar()
    plt.title('Electric potential [V]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    if show:
        plt.show()


class FiniteElementMethod:  # NOTE: only Electric Potential for now
    def __init__(self):
        self.mesh = None
        self.elements = None
        self.nodes = None
        self.S = None  # global matrix for electric potential
        self.V = None  # electric potential matrix
        self.Q = None  # boundary conditions matrix for electric potential
        self.global_is_built = False
        self.P = None  # power losses per element [W]
        self.E = None  # electric field per element
        self.triangulation = None
        self.triangles = None
        self.A = None  # global matrix for heat transfer
        self.T = None  # temperature matrix
        self.B = None  # boundary conditions matrix for heat transfer

    def set_elements(self, elements: List[Element]):
        self.elements = elements

    def set_nodes(self, nodes: List[Node]):
        self.nodes = nodes

    def open_mesh(self, path: str, material=None) -> None:
        """ Open msh mesh file. WARNING: this method overrides attributes 'elements' and 'nodes'
        Then, call method build_potential_global_matrix
        """
        if material is None:
            print('Warning: no material assigned to domain. Defaulting to air')
            material = air
        import meshreader
        self.mesh = meshreader.MeshReader(path)
        self.nodes = self.mesh.generate_nodes()
        self.elements = self.mesh.generate_elements(material=material)

        self._build_potential_global_matrix()

    def check_nodes(self):
        """ Checks if self.nodes was loaded """
        if self.nodes is None:
            raise AttributeError(f'No nodes were assigned to the FEM problem. '
                                 f'Call {self.open_mesh} to automatically load a mesh file (recommended)'
                                 f'or {self.set_nodes} to manually load the nodes')
        return True

    def check_elements(self):
        if self.nodes is None:
            raise AttributeError(f'No elements were assigned to the FEM problem. '
                                 f'Call {self.open_mesh} to automatically load a mesh file (recommended)'
                                 f'or {self.set_elements} to manually load the elements')
        return True

    def _build_potential_global_matrix(self):
        """ Builds matrix equation SV=Q (a.k.a. AX=B) for solving the electric potential distribuiton.
        The geometry and BC define S(A) and Q(B),
        and we wish to solve for V(X)"""

        if self.S is not None:
            print('Warning: Global matrix was already built and is now being overridden.')
        print('Bulding global matrix for electric potentials.')

        self.check_nodes()
        self.check_elements()

        self.S = np.zeros([len(self.nodes), len(self.nodes)])  # global elements matrix initialized
        self.Q = np.zeros(len(self.nodes))  # global bc matrix initialized

        for element in self.elements:
            s, q = element.get_elemental_matrix()
            for row in range(s.shape[0]):
                global_row = element.local2global(row)
                self.Q[global_row] += q[row]
                for col in range(s.shape[1]):
                    global_col = element.local2global(col)
                    # print(f'{element.id=}, {row=}, {col=}, {global_row=}, {global_col=}')  # for debugging
                    self.S[global_row, global_col] += s[row, col]
        
        # correction for duplicate nodes, which were causing the matrix to be singular:
        # force the self.S[n,n] = 1 for duplicated node with id n
        for i, node in enumerate(self.nodes):
            if not node.is_connected():
                n = node.id
                self.S[n, n] = 1
                print(f'Warning: Node {node.id} at {node.get_location()} is not connected. '
                      f'Global matrix corrected at [{n}, {n}].')
                
        self.global_is_built = True
        pass

    def apply_dirichlet(self,
                        node_id_list: Union[List[int], int],
                        value_list: Union[List[float], float]
                        ):
        """
        Apply dirichlet boundary conditions for the voltages (electric potential) at given nodes.
        :param node_id_list: int or list of ints.
        :param value_list: float or list of floats.
        If arguments are lists, they must have the same length
        :return: None
        """
        if not self.global_is_built:
            raise Warning(f'Global matrix must be built before applying boundary conditions! '
                          f'First call {self._build_potential_global_matrix}.')
        if isinstance(node_id_list, (int, float)):
            node_id_list = [node_id_list]
        if isinstance(value_list, (int, float)):
            value_list = [value_list]
        node_id_list = list(node_id_list)
        value_list = list(value_list)
        if len(value_list) != len(node_id_list):
            raise AttributeError(
                f'Length mismatch: node_id_list has length {len(node_id_list)}, '
                f'while value_list has length {len(value_list)}')

        for ibc in range(len(node_id_list)):  # ibc is the local index of nodes to have boundary conditions applied
            node_id = node_id_list[ibc]
            value = value_list[ibc]
            assert node_id in range(len(self.nodes)), f'Node {node_id} is not in range({len(self.nodes)})'
            N = 1e20  # any very large number
            self.S[node_id, node_id] = N
            self.Q[node_id] = N * value
        pass

    def solve_potential(self):
        """ Solves the linear system S*V = Q
        S: relates the relationships between the nodes
        V: vector of potentials we wish to solve for
        Q: depends on volumetric charge density per element
        """
        if self.S is not None:
            self.V = np.linalg.solve(self.S, self.Q)
            for p in range(self.V.shape[0]):
                self.nodes[p].potential = self.V[p]

        else:
            raise ValueError('Matrix S (a.k.a. A) not yet built!')

    def plot_all_nodes(self, markersize=25, show=False):
        """ Plots all instances of class Node assigned to instance of FiniteElementMethod """
        if not self.nodes:
            print(f'Warning: no instance of Node assigned to {self}')
            return
        for node in self.nodes:
            plt.plot(*node.get_location(), '.', markersize=markersize, c='k')
            x, y = node.get_location()
            plt.text(x, y, s=node.id, c='w', ha='center', va='center')
        if show:
            plt.show()

    def plot_all_elements(self, show=False, numbering=False, fill=False):
        """ Plots all instances of class Element assigned to instance of FiniteElementMethod """
        if fill:
            legend_patches = []
            for mat in Material.instances:
                color = mat.color
                color_name = mat.name
                patch = mpatches.Patch(color=color, label=color_name)
                legend_patches.append(patch)
            plt.legend(handles=legend_patches)

        if not self.elements:
            print(f'Warning: no instance of Element assigned to {self}')
            return
        for element in self.elements:
            plt.plot(*element.get_vertices(), c='black', linewidth=.1)
            if fill:
                color = element.material.color
                color_name = element.material.name
                plt.fill(*element.get_vertices(), color=color)

            if numbering:
                x, y = element.get_centroid()
                plt.text(x, y, s=str(element.id), c='k', ha='center', va='center')
        if show:
            plt.title('Geometry and materials.')
            plt.show()

    def triangulate(self):
        if self.triangulation is None:
            self.triangles = np.array([element.get_triangle() for element in self.elements])
            x = np.array([node.get_location()[0] for node in self.nodes])
            y = np.array([node.get_location()[1] for node in self.nodes])
            self.triangulation = tri.Triangulation(x, y, self.triangles)

    def plot_contour_potential(self, levels: int = 21, show: bool = False, cmap: str = 'jet'):
        """
        Plots the contour of the electric potential scalar field for nodes and elements defined in the global scope
        :param levels: int. Number of contour levels
        :param show: bool. Whether or not to show the plot (calls pyplot.show if True)
        :param cmap: str. String representing the cmap argument (colormap) of pyplot.tricontour
        :return: None
        """
        if levels < 2:
            raise ValueError('Contour plot needs at least 2 levels to be correctly displayed')

        self.triangulate()
        potentials = [node.potential for node in self.nodes]
        plt.tricontourf(self.triangulation, potentials, levels=levels-1, cmap=cmap)
        plt.colorbar()
        plt.title('Electric potential [V]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        if show:
            plt.show()

    def compute_E_field(self, _abs: bool = True, vector: bool = True, show: bool = False):
        """
        Compute and plot the Electric vector field quiver, which is the negative gradient of the electric potential.
        :param vector: bool. If True, plots the vector field quiver at each element.
        :param _abs: bool. If True, plots the scalar field of the absolute value at each element.
        :param show: bool. If True, calls pyplot.show at the end.
        :return: None
        """
        if self.V is not None:
            x, y = [], []  # centroids
            Ex, Ey = [], []  # orthogonal vector components
            triangles = []
            for element in self.elements:
                x0, y0 = element.get_centroid()
                x.append(x0)
                y.append(y0)

                ix, jy = element.get_E_field()
                Ex.append(ix)
                Ey.append(jy)
                element.E_abs = sqrt(ix**2 + jy**2)
                if _abs:
                    # node indexes that make up the element
                    triangles.append(element.get_triangle())
            self.E = Ex, Ey

            if _abs:
                # Old way of plotting the flat scalar fields. Works, but looks ugly (not smoothed)
                # nodes:
                # xp = np.array([node.get_location()[0] for node in self.nodes])
                # yp = np.array([node.get_location()[1] for node in self.nodes])
                #
                # absE = np.sqrt(np.array(Ex)**2 + np.array(Ey)**2)  # for each element
                # self.triangulate()
                # plt.tripcolor(xp, yp, facecolors=absE, triangles=triangles) # shading must be flat

                # Smoothed by interpolating the nodes while averaging neighboring elements into the nodes

                node_values = []
                for node in self.nodes:
                    node_values.append(node.average_from_elements('E_abs'))
                self.triangulate()

                ex = np.array([node.get_location()[0] for node in self.nodes])
                ey = np.array([node.get_location()[1] for node in self.nodes])
                z = tri.CubicTriInterpolator(self.triangulation, node_values, kind='min_E')(ex, ey)

                plt.tricontourf(self.triangulation, z)

                plt.colorbar()

            if vector:
                plt.quiver(x, y, Ex, Ey)

            if show:
                plt.title('Electric Field [V/m]')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.show()
            pass

        else:
            raise AttributeError(f'No solution has been found. First call {self.solve_potential}.')

    def compute_powerlosses(self, depth: Union[int, float], show: bool = True):
        """
        Computes self.P
        :param show: bool. If true, calls pyplot.show.
        :param depth: int or float. Depth of the extruded 2D domain.
        :return: float. Total power loss in the domain
        """
        self.depth = depth
        if self.V is None:
            raise AttributeError(f'No solution for potential found. First call {self.solve_potential}.')
        if self.E is None:
            raise AttributeError(f'The Electric Field E has not yet been calculated! First call {self.compute_E_field}.')
        self.P = []
        triangles = []
        for element in self.elements:
            Ex, Ey = element.get_E_field()
            power_loss = depth * element.area * element.material.cond_elect * (Ex ** 2 + Ey ** 2)
            element.power_loss = power_loss
            self.P.append(power_loss)
            triangles.append(element.get_triangle())
        self.P = np.array(self.P)

        # old way of plotting flat scalar field
        # xp = np.array([node.get_location()[0] for node in self.nodes])
        # yp = np.array([node.get_location()[1] for node in self.nodes])
        # plt.tripcolor(xp, yp, facecolors=self.P, triangles=triangles)  # shading must be flat

        node_values = []
        for node in self.nodes:
            node_values.append(node.average_from_elements('power_loss'))
        self.triangulate()

        x = np.array([node.get_location()[0] for node in self.nodes])
        y = np.array([node.get_location()[1] for node in self.nodes])
        z = tri.CubicTriInterpolator(self.triangulation, node_values, kind='min_E')(x, y)

        plt.tricontourf(self.triangulation, z, levels=20)

        plt.colorbar()
        total_loss = sum(self.P)
        if show:
            plt.title(f'Power losses [W] for geometry depth of {depth} m\n Total power loss in the domain: {total_loss:.3e} W')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.show()

        return total_loss
    
    def clear(self):  # FIXME
        """ Clears nodes and elements in global scope """
        Node.clear_instances()
        Element.clear_instances()

    # TODO: add solver for temperature distribution!

    """    
    def build_potential_global_matrix(self):
        ''' Builds matrix equation SV=Q (a.k.a. AX=B) for solving the electric potential distribuiton.
        The geometry and BC define S(A) and Q(B),
        and we wish to solve for V(X)'''

    """

    def _build_temperature_global_matrix(self, hbot, htop, Tamb):
        '''
        Builds matrix equation AT=B (a.k.a. AX=B) for solving the temperature distribution.
        The geometry and BC define A and B, and we wish to solve for T
        For each element, the equation is Pi - Pconv - Pcond = 0 (conservation of energy)
        '''
        # TODO: included outer boundaries heat flow. For now, only thin plates (depth) are valid  approximations
        if self.P is None:
            raise AttributeError(f'No power loss solution found. First call {self.compute_powerlosses}.')
        # TODO: copy potential matrix procedures here
        print('Bulding global matrix for temperature.')
        self.A = np.zeros([len(self.elements), len(self.elements)])  # global elements matrix initialized
        self.B = np.zeros(len(self.elements))  # global bc matrix initialized
        # ###
        for element in self.elements:
            # s depends only on the geometry, which is the same as the electric potential problem
            # s, _ = element.get_elemental_matrix()

            # # q, however, depends on the power generation of the element, instead of charge density
            # a = -self.P[element.id]/(element.material.cond_therm*self.depth*element.area)

            # q = a / 3 * np.array([[1], [1], [1]])
            # for row in range(s.shape[0]):
            #     global_row = element.local2global(row)
            #     for col in range(s.shape[1]):
            #         global_col = element.local2global(col)
            #         # print(f'{element.id=}, {row=}, {col=}, {global_row=}, {global_col=}')  # for debugging
            #         self.A[global_row, global_col] += s[row, col]
            row = element.id
            R_conv = 1/(element.area*(hbot+htop))  # convection resistance [K/W]
            Pi = element.power_loss
            self.A[row, row] += 1/R_conv
            self.B[row] = Pi + Tamb/R_conv

            for neighbor in element.get_neighbors():
                neighbor: Element
                col = neighbor.id

                diff = np.array(element.get_centroid()) - np.array(neighbor.get_centroid())
                distance = np.sqrt(diff[0]**2 + diff[1]**2)

                edge_nodes = tuple(set(element.nodes) & set(neighbor.nodes))
                edge_length = edge_nodes[0].distance_to(edge_nodes[1])
                # edge_length = np.sqrt(
                #     (edge_nodes[0].x - edge_nodes[1].x) +
                #     (edge_nodes[0].y - edge_nodes[1].y)
                # )
                sect_area = self.depth*edge_length  # depth * length of elements interface edge
                R_cond = distance/(element.material.cond_therm * sect_area) # conduction resistance [K/W]
                self.A[row, col] -= 1/R_cond
                self.A[row, row] += 1/R_cond




        # correction for duplicate nodes, which were causing the matrix to be singular:
        # force the self.A[n,n] = 1 for duplicated node with id n
        #for i, node in enumerate(self.nodes):
            #if not node.is_connected():
                #n = node.id
                #self.A[n, n] = 1
                #print(f'Warning: Node {node.id} at {node.get_location()} is not connected. '
                      #f'Global matrix corrected at [{n}, {n}].')
        # ###

    def solve_temperature(self, hbot, htop, Tamb=293):
        # TODO: read Temperature solver document
        """
        Computes the estimated temperature per element by applying conservation of energy:
        Assuming steady-state where dT/dt=0, the energy produced must be equal to the energy escaping
        T = Rth*P + T0
        :param Tamb: float. Ambient temperature. Defaults to 293 K (20ºC)
        :param hbot: float. Heat transfer coeff. [W/m²K] (conduction + convection) on the bottom (back) layer.
        :param htop: float. Heat transfer coeff. [W/m²K] (conduction + convection) on the top (front) layer.
        :return: None
        """


        """ Solves the linear system A*T = B
        A: relates the relationships between the nodes (geometry)
        T: vector of temperatures we wish to solve for
        B: depends on heat output per element
        :param T0: Union[float, int]. Initial temperature to override for every node
        """
        self.Tamb = Tamb
        if self.A is None:
            self._build_temperature_global_matrix(hbot, htop, Tamb)

        self.T = np.linalg.solve(self.A, self.B)
        for t in range(self.T.shape[0]):
            if Tamb is not None:
                self.elements[t].temperature = self.T[t]# + Tamb
            else:
                self.elements[t].temperature += self.T[t]

        for node in self.nodes:
            node.temperature = node.average_from_elements('temperature')


        #
        # # compute the equibibrium temperature (heating + convection):
        # for i in range(len(self.elements)):
        #     Rth = 1 / ( self.elements[i].area*(hbot + htop) )  # parallel thermal resistances [K/W]
        #     T = Rth * self.P[i] + Tamb
        #     self.elements[i].temperature = T
        # # node_temps = []
        # for node in self.nodes:
        #     # node_temps.append(node.average_from_elements('temperature'))
        #     node.temperature = node.average_from_elements('temperature')
        #
        # # compute heat diffusion:



    def plot_contour_temperature(self, levels: int = 21, show: bool = True, cmap: str = 'turbo'):
        """
        Plots the contour of the temperature scalar field for nodes and elements defined in the FEM scope
        :param levels: int. Number of contour levels
        :param show: bool. Whether or not to show the plot (calls pyplot.show if True)
        :param cmap: str. String representing the cmap argument (colormap) of pyplot.tricontour
        :return: None
        """
        #if False: # plot based on the nodes averaged temperatures
            #if levels < 2:
                #raise ValueError('Contour plot needs at least 2 levels to be correctly displayed')
            #self.triangulate()
            #temperatures = [node.temperature for node in self.nodes]
            #plt.tricontourf(self.triangulation, temperatures, levels=levels - 1, cmap=cmap)
            
            
        #else:
        xp = np.array([node.get_location()[0] for node in self.nodes])
        yp = np.array([node.get_location()[1] for node in self.nodes])
        
        # absE = np.sqrt(np.array(Ex)**2 + np.array(Ey)**2)  # for each element
        temp = np.array([e.temperature for e in self.elements])  # for each element
        self.triangulate()
        plt.tripcolor(xp, yp, facecolors=temp, triangles=self.triangles, cmap=cmap) # shading must be flat
            
        
        plt.colorbar()
        plt.title(f'Temperature [K]\nMax={round(max(temp),2)} K, '
                  f'Min={round(min(temp),2)} K, '
                  f'Avg={round(np.mean(temp),2)} K, '
                  f'ΔTmax={round(max(temp)-self.Tamb,2)} K')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        if show:
                plt.show()

def track_mouse():  # TEST
    def mouse_move(event):
        click = event.button
        x, y = event.xdata, event.ydata

        print(x, y, click)

    plt.connect('motion_notify_event', mouse_move)


if __name__ == '__main__':
    n0 = Node(0, 0)
    n1 = Node(0, 0.9)
    n2 = Node(0, 2.1)
    n3 = Node(-1.8, 1.6)
    n4 = Node(-2, 0.6)

    # Domain elements
    e0 = Element((n0, n1, n4), air)
    e1 = Element((n1, n3, n4), air)
    e2 = Element((n1, n2, n3), air)

    # element outside of domain
    offset = 1
    n5 = Node(0 + offset, 1 + offset)
    n6 = Node(0 + offset, 0 + offset)
    n7 = Node(1 + offset, 0 + offset)
    e3 = Element((n5, n6, n7), air)

    # neighbors
    print('Neighbors of Element 2:\n', e2.get_neighbors())

    n0.potential = 0
    n1.potential = 5
    n2.potential = 10
    n3.potential = 15
    n4.potential = 20
    n5.potential = 20

    global_plot_contour_potential(21)
    global_plot_all_nodes()
    global_plot_all_elements()

    plt.colorbar()
    plt.show()

    # TEST class FiniteElementMethod
    fem = FiniteElementMethod()
    fem.open_mesh('meshes/drilled_plate.msh')

    plt.show()
    pass
