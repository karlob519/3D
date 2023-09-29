import numpy as np
import numpy.linalg as LA
import pygame
from shapely.geometry import Point, Polygon

# Colours
black = (0, 0, 0)
white = (255, 255, 255)
orange = (255, 128, 10)
light_grey = (200, 200, 200)
shadow_colour = (100, 100, 100)
light_blue = (20, 150, 250)
dark_blue = (1, 47, 200)
bright_green = (10, 250, 50)
dark_green = (2, 120, 30)
red = (250, 0, 30)
dark_brown = (142, 92, 71)
light_brown = (240, 217, 181)
yellow = (250, 250, 5)
purple = (160, 50, 120)
bg_colour = light_grey

WIDTH, HEIGHT = 1200, 800
PI = np.pi
FOV = 120
origin = np.zeros(3)
x_vec, y_vec, z_vec = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])        

pyramid_check = {'1': '25342', '2': '13451', '3': '12451', '4': '13521', '5': '12431'}
octahedron_check = {'1': '25362', '2': '15461', '3': '15461', '4': '25362', '5': '12431', '6': '12431'}

# -------------------------  Classes  ---------------------------
class Environment:
    def __init__(self, start_POV: np.ndarray, FOV_vector=np.array([0, 0, -5]), FOV_angle=0):
        # Initialize the Environment object with a starting point of view (POV),
        # a field of view (FOV) vector, and FOV angle.
        self.objects = []                # List to store objects in the environment.
        self.POV = start_POV             # Starting point of view.
        self.FOV_vector = FOV_vector     # Direction vector of the field of view.
        self.FOV_angle = FOV_angle       # Angle of the field of view.

    def add(self, object):
        # Method to add objects to the environment.
        self.objects.append(object)

    def process(self):
        # Method to process the environment, sorting objects by their distance
        # from the view vector and then drawing them in reverse order.
        view_vec = self.POV + self.FOV_vector
        for object in sorted(self.objects, key=lambda x: LA.norm(x.pos - view_vec))[::-1]:
            # Draw objects based on specified draw_kwargs.
            object.draw(**object.draw_kwargs)

    def direction_changes(self, k: float = 1):
        # Calculate directional changes based on a scaling factor k.
        forward = k * self.FOV_vector
        FOV_vec_length = LA.norm(self.FOV_vector)
        up = np.array([0, FOV_vec_length, 0])
        left = k * np.cross(up, self.FOV_vector) / LA.norm(self.FOV_vector)
        up = k * up
        out_dict = {'forward': forward, 'back': -forward, 'up': up, 'down': -up, 'left': left, 'right': -left}
        return out_dict

    def view_vec(self):
        # Calculate the view vector by adding the POV and FOV vector.
        return self.POV + self.FOV_vector

    def translate(self, direction: str):
        # Translate the point of view (POV) in a specified direction.
        change = self.direction_changes(k=2)[direction]
        self.POV = self.POV + change

    def rotate_objects(self, direction: str, angle: float = 5):
        # Rotate objects in the environment around the POV based on a specified angle and direction.
        alpha = to_rad(angle)  # Convert angle to radians.
        if direction == 'horizontal':
            M = y_rotation(-alpha)  # Create a rotation matrix for horizontal rotation.
        elif direction == 'vertical':
            M = x_rotation(alpha)   # Create a rotation matrix for vertical rotation.

        for object in self.objects:
            # Apply the rotation matrix to each object's position relative to the POV.
            object.pos = rotate_point(M, object.pos, object.pos, self.POV)

    def rotate_FOV(self, angle: float, direction: str):
        # Rotate the field of view (FOV) vector based on a specified angle and direction.
        alpha = to_rad(angle)  # Convert angle to radians.
        if direction == 'up':
            M = x_rotation(alpha)    # Create a rotation matrix for upward FOV rotation.
        elif direction == 'down':
            M = x_rotation(-alpha)   # Create a rotation matrix for downward FOV rotation.
        elif direction == 'left':
            M = y_rotation(alpha)    # Create a rotation matrix for leftward FOV rotation.
        else:
            M = y_rotation(-alpha)   # Create a rotation matrix for rightward FOV rotation.

        # Apply the rotation matrix to the FOV vector.
        self.FOV_vector = rotate_point(M, self.FOV_vector, self.POV)

    def FOV_plane(self) -> np.ndarray:
        # Calculate the plane defined by the view vector and the FOV vector.
        return find_plane(self.view_vec(), self.FOV_vector)

    def FOV_normal(self) -> np.ndarray:
        # Calculate the normalized FOV vector.
        return self.FOV_vector / LA.norm(self.FOV_vector)

    def find_pygame_xy(self, point: np.ndarray) -> np.ndarray:
        # Convert a 3D point in the environment to 2D coordinates in a Pygame-like coordinate system.
        return find_xy(point, self.POV, self.FOV_vector)

    def find_2d_nodes(self, nodes: dict) -> dict:
        # Convert a dictionary of 3D nodes to 2D coordinates using the find_pygame_xy method.
        out_dict = {node: self.find_pygame_xy(pos) for node, pos in nodes.items()}
        return out_dict

    def __len__(self):
        # Return the number of objects in the environment.
        return len(self.objects)

    def __getitem__(self, ind):
        # Get an object from the environment by index.
        return self.objects[ind]

    
class Cuboid:
    def __init__(self, a: int, b: int, c: int, env: Environment=None):
        a1, b1, c1 = a/2, b/2, c/2
        self.name = 'cuboid'
        self.radius = np.sqrt(a1**2 + b1**2 + c1**2) 
        self.coords = [[-a1, -b1, -c1], [-a1, b1, -c1], [a1, -b1, -c1], [a1, b1, -c1],
                       [-a1, -b1, c1], [-a1, b1, c1], [a1, -b1, c1], [a1, b1, c1]]
        self.nodes = {str(ind+1): np.array(self.coords[ind]) for ind in range(8)}
        self.center = center_mass(self.nodes)
        self.edges = ['12', '15', '13', '24', '26', '34', '37', '48', '56', '57', '68', '78']
        self.surfaces = {'5678': red, '3478': purple, '2468': dark_blue, 
                        '1256': dark_green, '1234': orange, '1357': light_blue}
        self.ind_coords = [[-30, -40], [-30, 40], [30, 40], [30, -40]]
        self.env = env
    

    def find_not_shown(self, nodes: dict, debug: bool=False) -> list:
        pygame_coords = self.env.find_2d_nodes(nodes)
        dist_nodes = sorted(pygame_coords, key=lambda x: LA.norm(self.env.view_vec() - nodes[x]))
        dist_nodes.remove('center')
        test_nodes = dist_nodes[4:]
        output_list = [node for node in test_nodes if node_in_surfaces(node, self.surfaces.keys(), pygame_coords)]
        if debug is True:
            print(dist_nodes)
        return output_list
        
        
    def find_surfaces(self, nodes: dict):
        not_shown = self.find_not_shown(nodes)
        output_list = [key for key in self.surfaces if len(set(not_shown).intersection(set(key))) == 0]
        return output_list
    

    def draw_surfaces(self, nodes: dict, screen: pygame.Surface):
        surfaces =self.find_surfaces(nodes)
        pygame_coords = self.env.find_2d_nodes(nodes)
        for surface in surfaces:
            c1, c2, c3, c4 = surface
            p1, p2 = pygame_coords[c1], pygame_coords[c2]
            p3, p4 = pygame_coords[c3], pygame_coords[c4]
            colour = self.surfaces[surface]
            pygame.draw.polygon(screen, colour, (p1, p2, p4, p3))
        return
        
    
    def draw_shadow(self, shadow_points, surface):
        points = find_order(shadow_points)
        pygame.draw.polygon(surface, shadow_colour, points)
        return

    
    def show_edges(self, nodes: dict, debug: bool=False) -> list:
        not_shown = self.find_not_shown(nodes, debug)
        return [edge for edge in self.edges if len(set(not_shown).intersection(set(edge))) == 0]

        
class Pyramid:
    def __init__(self, a: int, b: int, c: int, env: Environment=None):
        self.name = 'pyramid'
        self.radius = np.sqrt(a**2 + b**2 + c**2) / 4
        self.coords = [[-a, -b, -c/5], [-a, b, -c/5], [a, -b, -c/5], [a, b, -c/5], [0, 0, 4*c/5]]
        self.nodes = {str(ind+1): np.array(self.coords[ind]) for ind in range(5)}
        self.node_list = ['1', '2', '3', '4', '5']
        self.edges = ['12', '13', '24', '34', '15', '25', '35', '45']
        self.surfaces = {'1234': red, '125': purple, '135': dark_blue, '245': orange, '345': dark_green}
        self.ind_coords = [[-30, -25], [30, -25], [0, 30]]
        self.env = env


    def point_in_shape(self, point: str, nodes: dict) -> bool:
        vectors = dict()
        pygame_coords = self.env.find_2d_nodes(nodes)
        end = pygame_coords[point]
        edge_str = pyramid_check[point]
        for p in edge_str[:4]:
            start = pygame_coords[p]
            vec = start - end
            vectors[p] = vec
        angle_sum = 0
        
        for i in range(4):
            p1, p2 = edge_str[i], edge_str[i+1]
            vec1, vec2 = vectors[p1], vectors[p2]
            angle = find_angle(vec1, vec2, in_deg=True)
            angle_sum += angle
        return abs(angle_sum-360) <= 2
    

    def find_not_shown(self, nodes: dict, debug: bool=False) -> str:
        for node in self.node_list:
            if self.point_in_shape(node, nodes):
                test_node = node
                break
        else: test_node = '0'
        
        furthest_node = max(nodes, key=lambda x: np.dot(self.env.FOV_vector, nodes[x]))
        if debug is True:
            print(f'Furthest: {furthest_node}, test: {test_node}')
        if furthest_node == test_node:
            return test_node
        else:
            return '0'
            
    
    def find_surfaces(self, nodes: dict) -> list:
        not_shown = self.find_not_shown(nodes)
        if not_shown != None:
            output_list = [key for key in self.surfaces if not_shown not in key]
        else:
            output_list = self.surfaces.keys()
        return output_list
    
    def z(self, nodes: dict) -> float:
        center = nodes['center']
        return -np.dot(self.env.FOV_vector, (nodes['5'] - center))


    def show_edges(self, nodes: dict, debug=False) -> list:
        pygame_coords = self.env.find_2d_nodes(nodes)
        end = pygame_coords['5']
        edge_list = [(pygame_coords[char[0]], pygame_coords[char[1]]) for char in self.edges[:4]]
        edges = dict(zip(self.edges[:4], edge_list))
        line_list = [(pygame_coords[char[0]], end) for char in self.edges[4:]]
        lines = dict(zip(self.edges[4:], line_list))
        crossers = []
        closest = min(nodes, key=lambda x: np.dot(self.env.FOV_vector, nodes[x]))
        closest_edge = '00'

        for point, edge in edges.items():
            for key, line in lines.items():
                if lines_intersect(edge, line):
                    if key not in crossers:
                        crossers.append(key)
                    closest_edge = point

        z = self.z(nodes)
        if debug is True:
            print(closest_edge, crossers, closest)
        
        if z < 0:
            output_list = [edge for edge in self.edges if edge not in crossers]
        else:
            if self.point_in_shape('5', nodes):
                not_shown_edges = ['00']
            else:
                not_shown_edges = crossers + [closest_edge]

            output_list = [edge for edge in self.edges if edge not in not_shown_edges]
        
        for edge in self.edges:
            if closest in edge and edge not in output_list:
                output_list.append(edge)
        if z >= 0 and self.find_not_shown(nodes) == '0':
            for edge in self.edges[4:]:
                if edge not in output_list:
                    output_list.append(edge)

        return output_list
    

    def draw_surfaces(self, nodes: dict, screen: pygame.Surface):
        surfaces = sorted(self.find_surfaces(nodes), key=lambda x: len(x))
        edges = [edge for edge in self.edges if edge not in self.show_edges(nodes)]
        pygame_coords = self.env.find_2d_nodes(nodes)
        z = self.z(nodes)
        for surface in surfaces:
            if z < 0:
                if len(surface) == 4:
                    c1, c2, c3, c4 = surface
                    p1, p2 = pygame_coords[c1], pygame_coords[c2]
                    p3, p4 = pygame_coords[c3], pygame_coords[c4]
                    colour = self.surfaces[surface]
                    pygame.draw.polygon(screen, colour, (p1, p2, p4, p3))
                else:
                    c1, c2, c3 = surface
                    p1, p2 = pygame_coords[c1], pygame_coords[c2]
                    if c1+c3 in edges or c2+c3 in edges:
                        None
                    else:
                        p3 = pygame_coords[c3]
                        colour = self.surfaces[surface]
                        pygame.draw.polygon(screen, colour, (p1, p2, p3))
            else:
                if len(surface) == 4:
                    None
                else:
                    c1, c2, c3 = surface
                    if c1 + c2 not in edges or self.point_in_shape('5', nodes):
                        p1, p2, p3 = pygame_coords[c1], pygame_coords[c2], pygame_coords[c3]
                        colour = self.surfaces[surface]
                        pygame.draw.polygon(screen, colour, (p1, p2, p3))
        return


    def draw_shadow(self, shadow_points, surface):
        points = find_order(shadow_points)
        pygame.draw.polygon(surface, shadow_colour, points)
        return


class Octahedron(Pyramid):
    def __init__(self, a:  int, b: int, c: int, env: Environment=None):
        self.name = 'octahedron'
        self.coords = [[-a, -b, 0], [-a, b, 0], [a, -b, 0], [a, b, 0], [0, 0, c], [0, 0, -c]]
        self.nodes = {str(ind+1): np.array(self.coords[ind]) for ind in range(6)}
        self.node_list = list(self.nodes.keys())
        self.center = center_mass(self.nodes)
        self.edges = ['12', '13', '24', '34', '15', '25', '35', '45', '16', '26', '36', '46']
        self.surfaces = {'125': purple, '135': dark_blue, '245': orange, '345': dark_green,
                         '126': red, '136': light_blue, '246': yellow, '346': bright_green}
        self.ind_coords = [[-30, 0], [0, 30], [30, 0], [0, -30]]
        self.env = env
    

    def point_in_shape(self, point: str, nodes: dict) -> bool:
        vectors = dict()
        pygame_coords = self.env.find_2d_nodes(nodes)
        end = pygame_coords[point]
        edge_str = octahedron_check[point]

        for p in edge_str[:4]:
            start = pygame_coords[p]
            vec = start - end
            vectors[p] = vec
        angle_sum = 0
        for i in range(4):
            p1, p2 = edge_str[i], edge_str[i+1]
            vec1, vec2 = vectors[p1], vectors[p2]
            angle = find_angle(vec1, vec2, in_deg=True)
            angle_sum += angle
        return abs(angle_sum-360) <= 2
    
    
    def find_not_shown(self, nodes: dict, debug: bool=False) -> str:
        hidden_nodes = []
        for node in self.node_list:
            if self.point_in_shape(node, nodes):
                hidden_nodes.append(node)
        if debug is True:
            print(hidden_nodes)

        if len(hidden_nodes) == 0:
            return None
        else:
            return max(hidden_nodes, key=lambda x: np.dot(self.env.FOV_vector, nodes[x]))

    
    def show_edges(self, nodes: dict, debug=False) -> list:
        pygame_coords = self.env.find_2d_nodes(nodes)
        end5, end6 = pygame_coords['5'], pygame_coords['6']
        edge_list = [(pygame_coords[char[0]], pygame_coords[char[1]]) for char in self.edges[:4]]
        edges = dict(zip(self.edges[:4], edge_list))
        line5_list = [(pygame_coords[char[0]], end5) for char in self.edges[4:]]
        lines5 = dict(zip(self.edges[4:], line5_list))
        line6_list = [(pygame_coords[char[0]], end6) for char in self.edges[4:]]
        lines6 = dict(zip(self.edges[4:], line6_list))
        crossers5 = []
        crossers6 = []
        closest_node = min(nodes, key=lambda x: np.dot(self.env.FOV_vector, nodes[x]))

        for point, edge in edges.items():
            for key, line in lines5.items():
                if lines_intersect(edge, line) and '5' in key:
                    if key not in crossers5:
                        crossers5.append(key)
                    closest_edge5 = point
            for key, line in lines6.items():
                if lines_intersect(edge, line) and '6' in key:
                    if key not in crossers6:
                        crossers6.append(key)
                    closest_edge6 = point

        z = self.z(nodes)
        
        if z < 0:
            if self.point_in_shape('6', nodes):
                not_shown_edge = '00'
            else:
                try:
                    not_shown_edge = closest_edge6
                except UnboundLocalError:
                    not_shown_edge = '00'
                if debug is True:
                    print(closest_edge6, crossers5)
            not_shown_list = crossers5 + [not_shown_edge]
            output_list = [edge for edge in self.edges if edge not in not_shown_list]
        else:
            if self.point_in_shape('5', nodes):
                not_shown_edge = '00'
            else:
                try:
                    not_shown_edge = closest_edge5
                except UnboundLocalError:
                    not_shown_edge = '00'
                if debug is True:
                    print(closest_edge5, crossers6)

            not_shown_list = crossers6 + [not_shown_edge]
            output_list = [edge for edge in self.edges if edge not in not_shown_list]

        not_shown_node = self.find_not_shown(nodes)
        if not_shown_node is not None:
            output_list = [edge for edge in output_list if not_shown_node not in edge]

        for edge in self.edges:
            if closest_node in edge and edge not in output_list:
                output_list.append(edge)

        return output_list
    

    def draw_surfaces(self, nodes: dict, screen: pygame.Surface):
        pygame_coords = self.env.find_2d_nodes(nodes)
        edges = self.show_edges(nodes)
        for surface, colour in self.surfaces.items():
            if edge_in_surf(edges, surface) is True:
                c1, c2, c3 = surface
                p1, p2, p3 = pygame_coords[c1], pygame_coords[c2], pygame_coords[c3]
                pygame.draw.polygon(screen, colour, [p1, p2, p3])


    def draw_shadow(self, shadow_points, surface):
        points = find_order(shadow_points)
        pygame.draw.polygon(surface, shadow_colour, points)
        return



# ------------------------- Functions ---------------------------
def center_mass(nodes: dict) -> np.ndarray:
    vec = np.array(list(nodes.values()))
    return np.mean(vec, axis=0)


def set_object(nodes: dict, position: np.ndarray) -> dict:
    out_dict = dict()
    for key, vector in nodes.items():
        out_dict[key] = vector + position
    
    return out_dict


def find_angle(vec1: np.ndarray, vec2: np.ndarray, in_deg: bool=True):
    ratio = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if in_deg is True:
        try:
            return round(np.arccos(ratio)*180/PI)  
        except ValueError:
            return 0
    else:
        try:
            return np.arccos(ratio) 
        except ValueError:
            return 0

    
def find_intersection(line1, line2):
    try:
        # Extract the coordinates of the endpoints of each line
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        # Calculate the determinants
        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Calculate the intersection point coordinates
        if det != 0:
            intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
            intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
            return intersection_x, intersection_y
        else:
            # Lines are parallel, no intersection
            return None
    except ValueError:
        return None
    

def lines_intersect(line1: list, line2: list):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Calculate the direction vectors of the lines
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    # Calculate the determinant
    determinant = dx1 * dy2 - dx2 * dy1

    # Check if the lines are parallel (determinant close to 0)
    if abs(determinant) < 1e-10:
        return False
    else:
        # Calculate the intersection point (if they are not parallel)
        t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / determinant
        t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / determinant

        # Check if the intersection point is within the line segments
        if 0 < t1 < 1 and 0 < t2 < 1:
            return True
        else:
            return False


def point_to_line_dist(point: tuple, line: list):
    x0, y0 = point
    (x1, y1), (x2, y2) = line

    # Calculate the coefficients of the line equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)

    # Calculate the distance
    distance = abs((A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2))
    
    return distance


def find_plane(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d = np.dot(point, normal)
    output_list = list(normal) + [d]
    return np.array(output_list)


def projection_on_plane(point_z: np.ndarray, plane: np.ndarray) -> np.ndarray:
    normal, d = plane[:3], plane[3]
    k = (d - np.dot(normal, point_z)) / (LA.norm(normal)**2)
    point_dash = point_z + k*normal

    return point_dash


def line_plane_intersection(line: list, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    start, end = line
    l_vector = end - start
    unit_vec = l_vector / LA.norm(l_vector)
    d = np.dot(point-start, normal) / np.dot(unit_vec, normal)
    return start + d*unit_vec


def str_in_str(str1: str, str2: str) -> bool:
    truth_vals = [char in str2 for char in str1]
    return sum(truth_vals) == len(str1)


def edge_in_surf(edges: list, surf: str) -> bool:
    e1, e2, e3 = surf[:2], surf[1:3], surf[0] + surf[2]
    return e1 in edges and e2 in edges and e3 in edges


def find_order(polygon: dict) -> list:
    n = len(polygon.keys()) - 1
    start_list = list(range(1, n+1))
    center = polygon['base']
    vectors = dict()
    for i in start_list:
        vec = polygon[str(i)] - center
        vectors[i] = vec

    node = start_list.pop()
    order_list = [str(node)]
    output_list = []

    while len(start_list) > 0:
        vec_n = vectors[node]
        next_node = min(start_list, key=lambda x: find_angle(vec_n, vectors[x]))
        order_list.append(str(next_node))
        node = next_node
        start_list.remove(next_node)
    
    for i in range(n):
        j, k = (i-1) % n, (i+1) % n
        n1, n2, n3 = order_list[j], order_list[i], order_list[k]
        line = [polygon[n1], polygon[n3]]
        vec = polygon[n2] - center
        if np.linalg.norm(vec) >= point_to_line_dist(center, line):
            output_list.append(polygon[n2])

    return output_list


# Rotations
def x_rotation(alpha: float) -> np.matrix:
    return np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])


def y_rotation(alpha: float) -> np.matrix:
    return np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]])


def z_rotation(alpha: float) -> np.matrix:
    return np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])


def rotate_2d(alpha: float) -> np.matrix:
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def rotate_point(M: np.matrix, point: np.ndarray, ref_point: np.ndarray, origin: np.ndarray=np.array([0, 0, 0])) -> np.ndarray:
    vector = ref_point - origin
    point_dash = point - vector
    new_point = np.dot(M, point_dash)
    return new_point + vector


def extend_line_3d(line: list, target: np.ndarray) -> np.ndarray:
    start, end = line
    start_vector = end - start
    unit_vec = start_vector / LA.norm(start_vector)
    vec_angle = find_angle(start_vector, target, in_deg=False)
    length = LA.norm(target - start) / np.cos(vec_angle)
    move_vec = unit_vec * length

    return start + move_vec


# Conversions
def to_rad(angle: float) -> float:
    return angle * PI / 180


def to_deg(angle: float) -> float:
    return round(angle*180/PI)


def to_pygame_coords(vector: np.ndarray, height: int=HEIGHT) -> np.ndarray:
    return np.array([vector[0], height - vector[1]])


def extend_line_3d(line: list, target: np.ndarray) -> np.ndarray:
    start, end = line
    start_vector = end - start
    unit_vec = start_vector / LA.norm(start_vector)
    vec_angle = find_angle(start_vector, target, in_deg=False)
    length = np.linalg.norm(target - start) / np.cos(vec_angle)
    move_vec = unit_vec * length

    return start + move_vec


def window(POV: np.ndarray, FOV_dir: np.ndarray, width: int=WIDTH, height: int=HEIGHT, angle: float=0) -> list:
    x_vec = np.array([width/2, 0, 0])
    y_vec = np.array([0, height/2, 0])
    if angle != 0:
        M = z_rotation(angle)
        x_vec = rotate_point(M, x_vec, origin)
        y_vec = rotate_point(M, y_vec, origin)
    
    view_vec = POV + FOV_dir
    normal = FOV_dir / LA.norm(FOV_dir)
    plane = find_plane(view_vec, normal)
    a, b = view_vec + x_vec, view_vec + y_vec

    x, y = projection_on_plane(a, plane), projection_on_plane(b, plane)

    e1, e2 = x - view_vec, y - view_vec
    e1 = e1 / LA.norm(e1)
    e2 = e2 / LA.norm(e2)

    return e1, e2


def find_xy(point_z, POV, FOV_dir):
    view_vec = POV + FOV_dir
    n = FOV_dir / LA.norm(FOV_dir)
    if np.dot(point_z-POV, n) < 0:
        return np.array([-1000, -1000])
    
    else:
        center = np.array([WIDTH/2, HEIGHT/2])
        z_dash = line_plane_intersection([POV, point_z], POV+FOV_dir, n)
        v1 = z_dash - view_vec
        e1, e2 = window(POV, FOV_dir)
        x, y = np.dot(v1, e1), np.dot(v1, e2)
        py_coords = to_pygame_coords(100*np.array([x, y]) + center)
        try:
            py_list = [round(x) for x in py_coords]
            return np.array(py_list)
        except ValueError:
            return np.array([-1000, -1000])

        

def point_in_shape(shape: list, test_point: np.ndarray, coords: dict) -> bool:
    shape_points = [coords[p] for p in shape]
    shape_points2 = shape_points[:2] + shape_points[2:][::-1]

    # Create a Polygon object from the shape_points
    poly1 = Polygon(shape_points)
    poly2 = Polygon(shape_points2)
    
    # Create a Point object from the test_point
    point = Point(test_point)
    
    # Check if the point is within the polygon
    return poly1.contains(point) or poly2.contains(point)


def node_in_surfaces(node: str, surfaces: list, coords: dict, k: int=9) -> bool:
    opp_node = str(k - int(node))
    shapes = [shape for shape in surfaces if opp_node in shape]
    point = coords[node]
    for surface in shapes:
        if point_in_shape(surface, point, coords):
            return True
    else:
        return False


if __name__ == '__main__':
    point = np.array([879, 485])
    shape = [np.array([858, 546]), np.array([840, 474]), np.array([905, 443]), np.array([927, 519]),]
    
    
