import numpy as np
import pygame
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
        

pyramid_check = {'1': '25342', '2': '13451', '3': '12451', '4': '13521', '5': '12431'}
octahedron_check = {'1': '25362', '2': '15461', '3': '15461', '4': '25362', '5': '12431', '6': '12431'}

direction_changes = {'up': (0, -10), 'down': (0, 10), 'left': (-10, 0), 'right': (10, 0)}

# ------------------------  Classes  -----------------------------------
class Environment:
    def __init__(self):
        self.objects = []
        self.POV = np.array([0, 0, 1000])
        self.space_pos = dict()
    

    def add(self, object):
        self.objects.append(object)
        space_p = list(object.pos) + [0]
        self.space_pos[object.num] = np.array(space_p)


    def process(self):
        for object in self.objects:
            object.pos = (self.space_pos[object.num] - self.POV)[:2]
            object.draw(**object.draw_kwargs)


    def move(self, direction):
        if direction == 'twist':
            for object in self.objects:
                object.rotate_points('z', 3)
        else:
            change_x, change_y = direction_changes[direction]
            vector = np.array([change_x, change_y, 0])
            for object in self.objects:
                self.space_pos[object.num] += vector

    
    def rotate(self, direction, angle=5):
        alpha = to_rad(angle)
        if direction == 'horizontal':
            M = y_rotation(-alpha)
        elif direction == 'vertical':
            M = x_rotation(alpha)

        for object in self.objects:
            vector = self.space_pos[object.num] - self.POV
            rot_vector = np.dot(M, vector)
            self.space_pos[object.num] = self.POV + rot_vector

    
    def rotate_POV(self, angle):
        alpha = to_rad(angle)
        M = z_rotation(alpha)
        self.POV = np.dot(M, self.POV)

        

    def __len__(self):
        return len(self.objects)
    

    def __getitem__(self, ind):
        return self.objects[ind]
    

class Cuboid:
    def __init__(self, a, b, c):
        a1, b1, c1 = a/2, b/2, c/2
        self.name = 'cuboid'
        self.radius = np.sqrt(a1**2 + b1**2 + c1**2) 
        self.coords = [[-a1, -b1, -c1], [-a1, b1, -c1], [a1, -b1, -c1], [a1, b1, -c1],
                       [-a1, -b1, c1], [-a1, b1, c1], [a1, -b1, c1], [a1, b1, c1]]
        self.nodes = {str(ind+1): np.array(self.coords[ind]) for ind in range(8)}
        self.center = center_mass(self.nodes)
        self.nodes['normal'] = np.array([0, 0, 1])
        self.edges = ['12', '15', '13', '24', '26', '34', '37', '48', '56', '57', '68', '78']
        self.surfaces = {'5678': red, '3478': purple, '2468': dark_blue, 
                        '1256': dark_green, '1234': orange, '1357': light_blue}
        self.ind_coords = [[-a/4, -b/4], [-a/4, b/4], [a/4, b/4], [a/4, -b/4]]

        
    def find_not_shown(self, show_list: bool=False):
        normal = self.nodes['normal']
        z = normal[2]
        dist_nodes = sorted(self.nodes.keys(), key=lambda x: dist(to_2d(self.center), to_2d(self.nodes[x])))
        dist_nodes.remove('normal')
        a, b = int(dist_nodes[0]), int(dist_nodes[1])
        if show_list is True:
            print(dist_nodes)
        if z >= 0:
            return str(min(a, b))
        else:
            return str(max(a, b))
        
        
    def find_surfaces(self):
        furthest_node = self.find_not_shown()
        output_list = [key for key in self.surfaces if furthest_node not in key]
        return output_list
    

    def draw_surfaces(self, position, screen):
        surfaces = self.find_surfaces()
        for surface in surfaces:
            c1, c2, c3, c4 = surface
            p1, p2 = to_2d(self.nodes[c1], position), to_2d(self.nodes[c2], position)
            p3, p4 = to_2d(self.nodes[c3], position), to_2d(self.nodes[c4], position)
            colour = self.surfaces[surface]
            pygame.draw.polygon(screen, colour, (p1, p2, p3, p4))
            pygame.draw.polygon(screen, colour, (p1, p3, p2, p4))
        
    
    def draw_shadow(self, shadow_points, surface):
        points = find_order(shadow_points)
        pygame.draw.polygon(surface, shadow_colour, points)
        return

    
    def show_edges(self):
        return self.edges

        
class Pyramid:
    def __init__(self, a, b, c):
        self.name = 'pyramid'
        self.radius = np.sqrt(a**2 + b**2 + c**2) / 4
        self.coords = [[-a, -b, -c/5], [-a, b, -c/5], [a, -b, -c/5], [a, b, -c/5], [0, 0, 4*c/5]]
        self.nodes = {str(ind+1): np.array(self.coords[ind]) for ind in range(5)}
        self.node_list = ['1', '2', '3', '4', '5']
        self.center = center_mass(self.nodes)
        self.nodes['normal'] = np.array([0, 0, 1])
        self.edges = ['12', '13', '24', '34', '15', '25', '35', '45']
        self.surfaces = {'1234': red, '125': purple, '135': dark_blue, '245': orange, '345': dark_green}
        self.ind_coords = [[-a/4, -c/4], [a/4, -c/4], [0, c/4]]


    def point_in_shape(self, point: str, shape: list):
        vectors = dict()
        end_x, end_y = to_2d(self.nodes[point])
        for p in shape:
            start_x, start_y = to_2d(self.nodes[p])
            vec = np.array([end_x - start_x, end_y - start_y])
            vectors[p] = vec
        angle_sum = 0
        edge_str = pyramid_check[point]
        for i in range(4):
            p1, p2 = edge_str[i], edge_str[i+1]
            vec1, vec2 = vectors[p1], vectors[p2]
            angle = find_angle(vec1, vec2)
            angle_sum += angle
        return abs(angle_sum-360) <= 2
    

    def find_not_shown(self, print_node: bool=False):
        normal = self.nodes['normal']
        test_node = '0'
        for node in self.node_list:
            shape = [el for el in self.node_list if el != node]
            if self.point_in_shape(node, shape):
                test_node = node
                break
        
        z = normal[2]
        if print_node is True:
            print(f'Node: {test_node}')
        
        if test_node == '0':
            return test_node
        elif test_node == '5':
            if z < 0:
                return test_node  
        else:
            if z > 0:
                return test_node
            
    
    def find_surfaces(self):
        not_shown = self.find_not_shown()
        if not_shown != None:
            output_list = [key for key in self.surfaces if not_shown not in key]
        else:
            output_list = self.surfaces.keys()
        return output_list


    def show_edges(self, print_edges=False):
        end = to_2d(self.nodes['5'])
        edge_list = [(to_2d(self.nodes[char[0]]), to_2d(self.nodes[char[1]])) for char in self.edges[:4]]
        edges = dict(zip(self.edges[:4], edge_list))
        line_list = [(to_2d(self.nodes[char[0]]), end) for char in self.edges[4:]]
        lines = dict(zip(self.edges[4:], line_list))
        crossers = []

        for nodes, edge in edges.items():
            for key, line in lines.items():
                if lines_intersect(edge, line):
                    if key not in crossers:
                        crossers.append(key)
                    closest_edge = nodes

        z = self.nodes['normal'][2]
        
        if z < 0:
            output_list = [edge for edge in self.edges if edge not in crossers]
        else:
            if self.point_in_shape('5', self.node_list[:4]):
                not_shown_edge = '00'
            else:
                try:
                    not_shown_edge = closest_edge
                except UnboundLocalError:
                    not_shown_edge = '00'
                if print_edges is True:
                    print(closest_edge, crossers)
            output_list = [edge for edge in self.edges if edge[0] not in not_shown_edge or edge[1] not in not_shown_edge]

        return output_list
    

    def draw_surfaces(self, position, screen):
        surfaces = sorted(self.find_surfaces(), key=lambda x: len(x))
        edges = [edge for edge in self.edges if edge not in self.show_edges()]
        z = self.nodes['normal'][2]
        for surface in surfaces:
            if z < 0:
                if len(surface) == 4:
                    c1, c2, c3, c4 = surface
                    p1, p2 = to_2d(self.nodes[c1], position), to_2d(self.nodes[c2], position)
                    p3, p4 = to_2d(self.nodes[c3], position), to_2d(self.nodes[c4], position)
                    colour = self.surfaces[surface]
                    pygame.draw.polygon(screen, colour, (p1, p2, p3, p4))
                    pygame.draw.polygon(screen, colour, (p1, p3, p2, p4))
                else:
                    c1, c2, c3 = surface
                    p1, p2 = to_2d(self.nodes[c1], position), to_2d(self.nodes[c2], position)
                    if c1+c3 in edges or c2+c3 in edges:
                        None
                    else:
                        p3 = to_2d(self.nodes[c3], position)
                        colour = self.surfaces[surface]
                        pygame.draw.polygon(screen, colour, (p1, p2, p3))
            else:
                if len(surface) == 4:
                    None
                else:
                    c1, c2, c3 = surface
                    if c1 + c2 not in edges or self.point_in_shape('5', self.node_list[:4]):
                        p1, p2 = to_2d(self.nodes[c1], position), to_2d(self.nodes[c2], position)
                        p3 = to_2d(self.nodes[c3], position)
                        colour = self.surfaces[surface]
                        pygame.draw.polygon(screen, colour, (p1, p2, p3))

    def draw_shadow(self, shadow_points, surface):
        points = find_order(shadow_points)
        pygame.draw.polygon(surface, shadow_colour, points)
        return


class Octahedron:
    def __init__(self, a, b, c):
        self.dimensions = [a, b, c]
        self.name = 'octahedron'
        self.radius = np.sqrt(a**2 + b**2 + c**2)
        self.coords = [[-a, -b, 0], [-a, b, 0], [a, -b, 0], [a, b, 0], [0, 0, c], [0, 0, -c]]
        self.nodes = {str(ind+1): np.array(self.coords[ind]) for ind in range(6)}
        self.node_list = list(self.nodes.keys())
        self.center = center_mass(self.nodes)
        self.nodes['normal'] = np.array([0, 0, 1])
        self.edges = ['12', '13', '24', '34', '15', '25', '35', '45', '16', '26', '36', '46']
        self.surfaces = {'125': purple, '135': dark_blue, '245': orange, '345': dark_green,
                         '126': red, '136': light_blue, '246': yellow, '346': bright_green}
        self.ind_coords = [[-a/4, 0], [0, c/4], [a/4, 0], [0, -c/4]]
    

    def point_in_shape(self, point: str, shape: list):
        vectors = dict()
        end_x, end_y = to_2d(self.nodes[point])
        for p in shape:
            start_x, start_y = to_2d(self.nodes[p])
            vec = np.array([end_x - start_x, end_y - start_y])
            vectors[p] = vec
        angle_sum = 0
        edge_str = octahedron_check[point]
        for i in range(4):
            p1, p2 = edge_str[i], edge_str[i+1]
            vec1, vec2 = vectors[p1], vectors[p2]
            angle = find_angle(vec1, vec2)
            angle_sum += angle
        return abs(angle_sum-360) <= 2
    
    
    def find_not_shown(self, print_node: bool = False):
        dist_nodes = sorted(self.nodes.keys(), key=lambda x: dist(to_2d(self.center), to_2d(self.nodes[x])))
        dist_nodes.remove('normal')
        node1, node2 = dist_nodes[:2]
        if print_node is True:
            print(f'Node1: {node1}, Node2: {node2}')
        if self.point_in_shape(node1, octahedron_check[node1]):
            return min(node1, node2, key=lambda x: self.nodes[x][2])
            
    
    def show_edges(self, print_edges=False):
        end5 = to_2d(self.nodes['5'])
        end6 = to_2d(self.nodes['6'])
        edge_list = [(to_2d(self.nodes[char[0]]), to_2d(self.nodes[char[1]])) for char in self.edges[:4]]
        edges = dict(zip(self.edges[:4], edge_list))
        line5_list = [(to_2d(self.nodes[char[0]]), end5) for char in self.edges[4:]]
        lines5 = dict(zip(self.edges[4:], line5_list))
        line6_list = [(to_2d(self.nodes[char[0]]), end6) for char in self.edges[4:]]
        lines6 = dict(zip(self.edges[4:], line6_list))
        crossers5 = []
        crossers6 = []

        for nodes, edge in edges.items():
            for key, line in lines5.items():
                if lines_intersect(edge, line) and '5' in key:
                    if key not in crossers5:
                        crossers5.append(key)
                    closest_edge5 = nodes
            for key, line in lines6.items():
                if lines_intersect(edge, line) and '6' in key:
                    if key not in crossers6:
                        crossers6.append(key)
                    closest_edge6 = nodes

        z = self.nodes['normal'][2]
        
        if z < 0:
            if self.point_in_shape('6', self.node_list[:4]):
                not_shown_edge = '00'
            else:
                try:
                    not_shown_edge = closest_edge6
                except UnboundLocalError:
                    not_shown_edge = '00'
                if print_edges is True:
                    print(closest_edge6, crossers5)
            not_shown_list = crossers5 + [not_shown_edge]
            output_list = [edge for edge in self.edges if edge not in not_shown_list]
        else:
            if self.point_in_shape('5', self.node_list[:4]):
                not_shown_edge = '00'
            else:
                try:
                    not_shown_edge = closest_edge5
                except UnboundLocalError:
                    not_shown_edge = '00'
                if print_edges is True:
                    print(closest_edge5, crossers6)

            not_shown_list = crossers6 + [not_shown_edge]
            output_list = [edge for edge in self.edges if edge not in not_shown_list]

        not_shown_node = self.find_not_shown()
        if not_shown_node is not None:
            output_list = [edge for edge in output_list if not_shown_node not in edge]

        return output_list
    

    def draw_surfaces(self, position, screen):
        edges = self.show_edges()
        for surface, colour in self.surfaces.items():
            if edge_in_surf(edges, surface) is True:
                c1, c2, c3 = surface
                p1, p2, p3 = to_2d(self.nodes[c1], position), to_2d(self.nodes[c2], position), to_2d(self.nodes[c3], position)
                pygame.draw.polygon(screen, colour, [p1, p2, p3])


    def draw_shadow(self, shadow_points, surface):
        points = find_order(shadow_points)
        pygame.draw.polygon(surface, shadow_colour, points)
        return


# ------------------------- Functions ---------------------------
def to_2d(vector, center=(0, 0)):
    x, y = center
    return x + vector[0], y - vector[1]


def dist(x: tuple, y: tuple):
    return (x[0] - y[0])**2 + (x[1] - y[1])**2


def center_mass(nodes: dict):
    vec = np.array(list(nodes.values()))
    return np.mean(vec, axis=0)


def find_angle(vec1: np.ndarray, vec2: np.ndarray, in_deg: bool=True):
    ratio = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if in_deg is True:
        try:
            return round((np.arccos(ratio))*180/np.pi)  
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


def str_in_str(str1: str, str2: str) -> bool:
    truth_vals = [char in str2 for char in str1]
    return sum(truth_vals) == len(str1)


def edge_in_surf(edges: list, surf: str) -> bool:
    e1, e2, e3 = surf[:2], surf[1:3], surf[0] + surf[2]
    return e1 in edges and e2 in edges and e3 in edges


def get_vector(start_point: tuple, end_point: tuple) -> np.ndarray:
    x1, y1 = start_point
    x2, y2 = end_point
    return np.array([x2-x1, y2-y1])


def find_order(polygon: dict) -> list:
    n = len(polygon.keys())-1
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


# Conversions
def to_rad(angle: float) -> float:
    return angle * np.pi / 180


def to_deg(angle: float) -> float:
    return round(angle*180/np.pi)


def extend_line_3d(line: list, target: np.ndarray) -> np.ndarray:
    start, end = line
    start_vector = end - start
    unit_vec = start_vector / np.linalg.norm(start_vector)
    vec_angle = find_angle(start_vector, target, in_deg=False)
    length = np.linalg.norm(target - start) / np.cos(vec_angle)
    move_vec = unit_vec * length

    return start + move_vec

if __name__ == '__main__':
    p1 = np.array([1, 2, 1])
    p2 = np.array([5, 3, 7])
    target = np.array([0, 0, 10])
    print(extend_line_3d([p1, p2], target))










