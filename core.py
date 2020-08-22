# Author: Eduardo Eller Behr
#
#
#
# Reference: Ida, Nathan. 'Engineering Electromagnetics', 3rd Ed. Springer Verlag
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from numpy import sqrt, mean, ndarray, matrix
from typing import Tuple, List, Union


def plot_all_nodes(markersize=25, show=False):
    if not Node.instances:
        print('Warning: no instance of Node found')
        return
    for node in Node.instances:
        plt.plot(*node.get_location(), '.', markersize=markersize, c='k')
        x, y = node.get_location()
        x_offset, y_offset = (0.0, 0.0)  # offsets
        plt.text(x + x_offset, y + y_offset, s=node.id, c='w', ha='center', va='center')
    if show:
        plt.show()


class Material:
    def __init__(self, name: str, electrical_conductivity: float, thermal_conductivity: float, permittivity: float = 1,
                 rgb: Tuple[float,float,float] = (1, 1, 1)):
        self.name = name
        self.cond_elect = electrical_conductivity  # Siemens
        self.cond_therm = thermal_conductivity
        self.permittivity = permittivity
        if len(rgb) != 3:
            raise ValueError('Parameter rgb must be a tuple of length 3.')
        self.color = rgb

    def __repr__(self):
        return f"""{self.name}:
        Electrical Conductivity: {self.cond_elect} [Siemens]
        Thermal Conductivity: {self.cond_therm} [UNIT]"""
    pass


copper = Material('Copper', 1e7, 10, 1000, rgb=(1., 128 / 255, 0.))
air = Material('Air', 1e-9, 1, 1, rgb=(206 / 255, 1., 1.))


# class Element:  # stub
#     pass


class Node:
    instances = []
    count = 0  # counts the instances
    id_dict = {}

    @classmethod
    def inc(cls):
        """ increments class variable count """
        cls.count += 1

    @classmethod
    def update_id(cls, dictionary):
        cls.id_dict.update(dictionary)

    @classmethod
    def get_node(cls, id):
        """ Returns Node object whose id is given """
        # ref = {obj:obj.id for obj in instances}
        return cls.id_dict[id]

    def __init__(self, x: int, y: int, init_temp=0, init_potential=0, lock_temp=False, lock_potent=False):
        self.x = x
        self.y = y
        self.temperature = init_temp
        self.potential = init_potential
        self.temperature_locked = lock_temp
        self.potential_locked = lock_potent
        self.neighbors = {'North': None, 'South': None, 'East': None, 'West': None}
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

    def __repr__(self):
        return f"\nNode {self.id} at {self.get_location()}. Temp: {self.get_temperature()} K. Potential: {self.potential}V"

    def set_neighbor(self, **kwargs):
        for kw in kwargs:
            if kw in self.neighbors.keys():
                self.neighbors.update(kwargs)
            else:
                raise KeyError(f'Unknown key. Try any of these: {self.neighbors.keys().__str__()[10:-1]}')

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

    def isNode(self, obj, type_error=True):
        """ Checks whether obj is an instanceof this class """
        if isinstance(obj, self.__class__):
            return True
        else:
            if type_error:
                raise TypeError(f'{obj} is no instance of {self.__class__}')
            return False

    pass


class Element:
    instances = []
    id_dict = {}  # dictionary of id
    count = 0  # counts the instances

    @classmethod
    def inc(cls):
        """ increments class variable count """
        cls.count += 1

    @classmethod
    def get_element(cls, id):
        """ Returns Element object whose id is given """
        # ref = {obj:obj.id for obj in instances}
        return cls.id_dict[id]

    def __init__(self, nodes: Tuple['Node'], material: Material, ρ: float = 0):
        """
        Creates a triangular element
        :param nodes: Tuple['Node']
        :param material: Material
        """
        if not isinstance(nodes, (tuple, list, ndarray)): raise TypeError('Parameter nodes must be a tuple of 3 nodes')
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(f'Object {node} is not instance of {Node}')
        self.nodes = tuple(nodes)
        if len(self.nodes) != 3: raise ValueError("An element must be triangular (exactly 3 nodes)")
        self.ρ = ρ  # volumetric charge density
        self.Δ = self.get_area()  # area of the element (triangular)

        self.assign_nodes_to_element()
        self.material = material
        self.instances.append(self)
        self.id = self.count
        self.id_dict.update({self.id: self.instances})
        self.inc()
        pass

    def __repr__(self) -> str:
        return f"Element {self.id} with nodes {[node.id for node in self.nodes]} and area {np.round(self.Δ, 6)}.\n"

    def set_charge_density(self, value: float):
        self.ρ = float(value)

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

    def get_vertices(self) -> Tuple[float, float]:
        """ Returns the coordinates of all nodes, with the first replicated """
        x = [node.get_location()[0] for node in self.nodes];
        x.append(x[0])
        y = [node.get_location()[1] for node in self.nodes];
        y.append(y[0])
        return x, y

    def get_triangle(self) -> List[int]:
        """ Returns a list of 3 nodes id to reference the triangle vertices """
        return [self.nodes[i].id for i in range(3)]

    def get_centroid(self) -> Tuple[float, float]:
        """ Returns the (x,y) coordinates of the centroid of a triagle (mean of each axis' coordinates points) """
        x_coord = [node.get_location()[0] for node in self.nodes]
        y_coord = [node.get_location()[1] for node in self.nodes]
        x_cent = mean(x_coord)
        y_cent = mean(y_coord)
        return x_cent, y_cent

    def get_area(self) -> float:
        """ Returns the area of the triangle computed with the vertices coordinates """
        x0, x1, x2 = [self.nodes[i].x for i in range(3)]
        y0, y1, y2 = [self.nodes[i].y for i in range(3)]
        return abs((x0 - x2) * (y1 - y0) - (x0 - x1) * (y2 - y0)) / 2

    def get_elemental_matrix(self) -> Tuple[matrix, matrix]:
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

        # TODO OPTIMIZATION: remove redundant assignments and put them directly in the matrix
        s = np.matrix([[s11, s12, s13],
                       [s21, s22, s23],
                       [s31, s32, s33]])

        q = self.ρ * self.Δ / 3 * matrix([1, 1, 1]).transpose()
        return s, q

    def local2global(self, local_index: int) -> int:
        """ Converts local index to global index
        :param local_index: int. Number 0, 1 or 2 (i, j, k) used to refer to the vertices of the element
        :return global_index: int. Number 0:n used to refer to all the vertices of the whole domain, where n-1 is the total number of vertices (nodes).
        """
        if local_index in [0, 1, 2]:
            global_index = self.nodes[local_index].id
            return global_index
        else:
            raise ValueError('Parameter local_index must be 0, 1 or 2')

    def global2local(self, global_index: int) -> int:
        """ Converts global index to local index
        :param global_index: int. Number 0:n used to refer to all the vertices of the whole domain, where n-1 is the total number of vertices (nodes).
        :return local_index: int. Number 0, 1 or 2 that refers to the local vertex of the element
        """
        _node = Node.instances[global_index]
        if _node in self.nodes:
            local_index = self.nodes.index(_node)
            return local_index
        else:
            raise ValueError(f'Node {_node.id} is not part of element {self.id}')

    def get_E_field(self):
        """ Returns the negative of the gradient of the potential """
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

        partial_x = sum([q[l] * V[l] for l in [i, j, k]]) / (2 * self.get_area())
        partial_y = sum([r[l] * V[l] for l in [i, j, k]]) / (2 * self.get_area())

        return -partial_x, -partial_y


def plot_all_elements(show=False, numbering=True, fill=False):
    """ Plots all instances of class Element """
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
    if show: plt.show()


def plot_contour_potential(levels=21, show=False, cmap='jet'):
    elements = Element.instances
    triangles = [element.get_triangle() for element in elements]
    nodes = Node.instances
    x = np.array([node.get_location()[0] for node in nodes])
    y = np.array([node.get_location()[1] for node in nodes])
    triangulation = tri.Triangulation(x, y, triangles)
    potentials = [node.potential for node in nodes]
    plt.tricontourf(triangulation, potentials, levels=levels, cmap=cmap)
    plt.colorbar()
    plt.title('Electric potential [V]')
    plt.xlabel('x [m]');
    plt.ylabel('y [m]')
    if show: plt.show()


class FiniteElementsMethod:  # NOTE: only Electric Potential for now
    def __init__(self, elements: List['Elements'] = [], nodes: List['Nodes'] = []):
        self.elements = elements
        self.nodes = nodes
        self.S = np.matrix([])
        self.V = np.matrix([])
        self.Q = np.matrix([])
        self.global_is_built = False

    def open_mesh(self, path: str) -> None:
        """ Open msh mesh file. WARNING: this method overides attributes 'elements' and 'nodes'"""
        import meshreader
        self.mesh = meshreader.MeshReader(path)
        self.nodes = self.mesh.generate_nodes()
        self.elements = self.mesh.generate_elements()

    def build_global_matrix(self):
        """ Builds matrix equation SV=Q (a.k.a. AX=B). The geometry and BC define S(A) and Q(B), and we wish to solve for V(X)"""
        self.S = np.zeros([len(self.nodes), len(self.nodes)])  # global elements matrix initialized
        self.Q = np.zeros(len(self.nodes))  # global bc matrix initialized

        for element in self.elements:  # TODO: change reference from global instances to FEM object attribute 'self.elements'
            s, q = element.get_elemental_matrix()
            for row in range(s.shape[0]):
                global_row = element.local2global(row)
                self.Q[global_row] += q[row]
                for col in range(s.shape[1]):
                    global_col = element.local2global(col)
                    # print(f'{element.id=}, {row=}, {col=}, {global_row=}, {global_col=}')

                    self.S[global_row, global_col] += s[row, col]
        self.global_is_built = True
        pass

    def apply_bc(self, node_id_list: Union[List, int], value_list: Union[List[float], float]):
        """ Apply boundary conditions """
        if not self.global_is_built:
            raise Warning('Global matrix must be built before applying boundary conditions!')
        if isinstance(node_id_list, (int, float)):
            node_id_list = [node_id_list]
        if isinstance(value_list, (int, float)):
            value_list = [value_list]
        node_id_list = list(node_id_list)
        value_list = list(value_list)
        if len(value_list) != len(node_id_list):
            raise AttributeError(
                f'Length mismatch: node_id_list has length {len(node_id_list)}, while value_list has length {len(value_list)}')

        for ibc in range(len(node_id_list)):
            node_id = node_id_list[ibc]
            value = value_list[ibc]
            assert node_id in range(len(self.nodes)), f'Node {node_id} is not in range({len(self.nodes)})'
            N = 1e20  # a very large number
            self.S[node_id, node_id] = N
            self.Q[node_id] = N * value
        pass

    def solve(self):
        if self.S.any():
            self.V = np.linalg.solve(self.S, self.Q)
            for p in range(self.V.shape[0]):
                self.nodes[p].potential = self.V[p]

        else:
            raise ValueError('Matrices S(A) not yet built!')

    def plot_gradient(self, show=False):
        if self.V is not None:
            x, y = [], []
            Ex, Ey = [], []
            for element in self.elements:
                x0, y0 = element.get_centroid()
                x.append(x0);
                y.append(y0)

                ix, jy = element.get_E_field()
                Ex.append(ix);
                Ey.append(jy)
            plt.quiver(x, y, Ex, Ey)
            if show: plt.show()
            pass

        else:
            raise AttributeError('No solution has been found. Solve the problem first!')

    def plot(self):
        if self.V:
            ...
        else:
            raise AttributeError('No solution has been found. Solve the problem first!')
        pass


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

    n0.potential = 0
    n1.potential = 5
    n2.potential = 10
    n3.potential = 15
    n4.potential = 20
    n5.potential = 20

    plot_contour_potential(21)

    plot_all_nodes()
    plot_all_elements()
    plt.colorbar()
    plt.show()

    # TEST class FiniteElementsMethod
    fem = FiniteElementsMethod()
    fem.open_mesh('meshes/semicircle.msh')
    pass
