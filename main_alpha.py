# Imports
import pygame
import numpy as np
import sys
import backend_alpha

# Display
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1800, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Cube')
calibri = pygame.font.SysFont('Calibri', 20)
ariel = pygame.font.SysFont('Ariel', 30)
node_font = pygame.font.SysFont('Ariel', 50)

# Clock
clock = pygame.time.Clock()
FPS = 30

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

screen.fill(bg_colour)

# Constants
PI = np.pi
rotations = {'up': ('x', -3), 'down': ('x', 3), 'right': ('y', 3), 'left': ('y', -3), 'twist': ('z', 3)}

# Shapes
a, b, c = 75, 150, 100
CUBOID = backend_alpha.Cuboid(a, b, c)
CUBE = backend_alpha.Cuboid(c, c, c)
PYRAMID = backend_alpha.Pyramid(100, 75, 150)
OCTAHEDRON = backend_alpha.Octahedron(100, 100, 144)
CUBE.name = 'cube'


env = backend_alpha.Environment()

light_source = np.array([WIDTH/2, 1000, 0])

x_vec = np.array([100, 0, 0])
y_vec = np.array([0, 100, 0])
z_vec = np.array([0, 0, 100])
xyz_vecs = [x_vec, y_vec, z_vec]

# Commands
left1, left2 = f'{chr(8592)} : rotate left', f'{chr(8592)} : move left'
right1, right2 = f'{chr(8594)} : rotate right', f'{chr(8594)} : move right'
up1, up2 = f'{chr(8593)} : rotate up', f'{chr(8593)} : move up'
down1, down2 = f'{chr(8595)} : rotate down', f'{chr(8595)} : move down'
t = f't : twist'
one = '1 : toggle surfaces'
two = '2 : toggle nodes'
tab = 'TAB : toggle shadows'
commands1 = [left1, right1, up1, down1, t, one, two, tab]
commands2 = [left2, right2, up2, down2, t, one, two, tab]

# ---------------------  Classes  ------------------------ 
class Object:
    def __init__(self, shape, position=[WIDTH/2, HEIGHT/2], environment=env):
        self.shape = shape
        self.nodes = self.shape.nodes
        self.pos = position
        self.axis = Axis(self.pos)
        self.num = len(env)
        self.draw_kwargs = {'edge_colour': black, 'show_surfaces': True, 'show_nodes': False, 'shadow': False,
               'axis': False, 'show_edges': True}
        environment.add(self)


    def rotate_points(self, direction: str, angle: float=5):
        alpha = to_rad(angle)
        if direction == 'x':
            M = x_rotation(alpha)
        elif direction == 'y':
            M = y_rotation(alpha)
        else:
            M = z_rotation(alpha)
        
        for node, vector in self.nodes.items():
            self.nodes[node] = rotate_point(M, vector)
        
        self.axis.rotate_points(direction, angle)
        

    def boundary(self):
        centre = to_2d(self.shape.center, self.pos)
        points = sorted(self.nodes, key=lambda x: dist(to_2d(self.nodes[x], self.pos), centre))[-4:]
        return ''.join(points)

    
    def shadow(self, bottom=HEIGHT-120, source_pos=light_source):
        start_pos = to_2d(source_pos)
        xy_coords = {node: to_2d(pos, self.pos) for node, pos in self.nodes.items() if node != 'normal'}
        normal_y = self.pos[1]
        delta_y = bottom-normal_y
        shadow_dict = {'base': extend_line(start_pos, self.pos, delta_y)}
        for node, pos in xy_coords.items():
            adjustment = normal_y - pos[1]
            end_pos = extend_line(start_pos, pos, delta_y + 0.5*adjustment)
            shadow_dict[node] = end_pos

        return shadow_dict


    def draw(self, edge_colour=black, show_nodes: bool=False, show_surfaces:bool=False, shadow: bool=False, axis: bool=False, show_edges: bool=False):
        not_shown = self.shape.find_not_shown()
        if show_nodes is True:
            for node, vector in self.nodes.items():
                if node != 'normal':    
                    x, y = to_2d(vector, self.pos)
                    if node != not_shown:
                        pygame.draw.circle(screen, red, (x, y), 10)
                        str_val = node
                        val = node_font.render(str_val, True, light_blue)
                        screen.blit(val, (x, y))

        if shadow is True:
            shadow_points = self.shadow(source_pos=light_source)
            self.shape.draw_shadow(shadow_points, screen)
            
        if axis is True:
            self.axis.draw()

        if show_surfaces is True:
            self.shape.draw_surfaces(self.pos, screen)

        if show_edges is True:
            for pair in self.shape.edges:
                if pair in self.shape.show_edges():
                    p1, p2 = pair[0], pair[1]
                    if p1 != not_shown and p2 != not_shown:
                        node1, node2 = to_2d(self.nodes[p1], self.pos), to_2d(self.nodes[p2], self.pos)
                        draw_line(node1, node2, edge_colour)

        return
    

class Axis:
    def __init__(self, center):
        self.center = center
        self.vectors = [x_vec, y_vec, z_vec]
        self.angles = [0, 0, 0]
    
    def rotate_points(self, direction: str, angle: float=5):
        alpha = to_rad(angle)
        if direction == 'x':
            M = x_rotation(alpha)
        elif direction == 'y':
            M = y_rotation(alpha)
        else:
            M = z_rotation(alpha)
        
        for i in range(3):
            vector = self.vectors[i]
            self.vectors[i] = rotate_point(M, vector)
            self.angles[i] = find_angle(self.vectors[i], xyz_vecs[i])
            
        
    def draw(self, colour=black, pos=(0, 0)):
        x, y, z = self.angles
        angles_str = f'X: {x}   Y: {y}   Z: {z}'
        angles_val = ariel.render(angles_str, True, colour)
        screen.blit(angles_val, pos)
            
    
# ------------------------  Functions  -------------------------
def draw_line(start_point: tuple, end_point: tuple, colour: tuple, surface=screen, flip: bool=False):
    pygame.draw.line(surface, colour, start_point, end_point, width=2)
    if flip:
        pygame.display.flip()
    return


def draw_polygon(point_list: list, pos: tuple, colour: tuple, surface: pygame.Surface=screen, fill: bool=False):
    w = 0 if fill is True else 1
    points_to_draw = [to_2d(point, pos) for point in point_list]
    pygame.draw.polygon(surface, colour, points_to_draw, width=w)

    return


def extend_line(point1:tuple, point2: tuple, delta_y: float) -> tuple:
    x1, y1 = point1
    x2, y2 = point2
    a = (y2-y1) / (x2-x1) if x2 != x1 else 1 
    x = (y2 + delta_y - y1) / a + x1
    return np.array([x, y2 + delta_y])


def to_rad(angle: float) -> float:
    return angle * PI / 180


def to_deg(angle: float) -> float:
    return round(angle*180/PI)


def to_2d(vector: np.ndarray, center=(0, 0)) -> tuple:
    x, y = center
    return x + vector[0], y - vector[1]


def x_rotation(alpha: float) -> np.matrix:
    return np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])


def y_rotation(alpha: float) -> np.matrix:
    return np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]])


def z_rotation(alpha: float) -> np.matrix:
    return np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])


def rotate_point(M: np.matrix, v: np.ndarray) -> np.ndarray:
    return np.dot(M, v)


def dist(x: tuple, y: tuple) -> float:
    return (x[0] - y[0])**2 + (x[1] - y[1])**2


def in_range(object: Object, point: tuple, boundary: str) -> bool:
    vectors = dict()
    end_x, end_y = point
    for p in boundary:
        start_x, start_y = to_2d(object.nodes[p], object.pos)
        vec = np.array([end_x - start_x, end_y - start_y])
        vectors[p] = vec

    angle_sum = 0
    edge_str = boundary + boundary[0]
    for i in range(4):
        p1, p2 = edge_str[i], edge_str[i+1]
        vec1, vec2 = vectors[p1], vectors[p2]
        angle = find_angle(vec1, vec2)
        angle_sum += angle

    return angle_sum-360 >= 0


def find_angle(vec1: tuple, vec2: tuple) -> float:
    ratio = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return to_deg(np.arccos(ratio))



def object_choice(pos: tuple, env: list=env) -> Object:
    for obj in env:
        boundary = obj.boundary()
        if in_range(obj, pos, boundary) is True:
            return obj
    else:
        return None
    

def draw_indicators(curr_obj: Object):
    curr_num = curr_obj.num if curr_obj is not None else None
    n = len(env)
    for i in range(n):
        obj = env[i]
        pos = (i*75 + 30, HEIGHT-50)
        if i == curr_num:
            draw_polygon(obj.shape.ind_coords, pos, red, fill=True)
        else:
            draw_polygon(obj.shape.ind_coords, pos, black)
    
    return


def change_draw_kwargs(curr_obj: Object, key: str):
    global env
    if curr_obj is None:
        for obj in env:
            obj.draw_kwargs[key] = not obj.draw_kwargs[key]
    else:
        curr_obj.draw_kwargs[key] = not curr_obj.draw_kwargs[key]
    
    return 
        

def show_commands(comm_list, colour=black):
    n = len(comm_list)
    for i in range(n):
        pos = (20, 25 + i*22)
        text_str = comm_list[i]
        val = calibri.render(text_str, True, colour)
        screen.blit(val, pos)


# ----------------------  Objects  ----------------------------
x, y = 600, 200

cuboid = Object(CUBOID, [x, y])
cuboid.rotate_points('x', 10)
cuboid.rotate_points('y', 16)

pyramid = Object(PYRAMID, [x+300, y])
pyramid.rotate_points('x', 20)
pyramid.rotate_points('y', 20)

cube = Object(CUBE, [x, y+300])
cube.rotate_points('x', 20)
cube.rotate_points('y', 20)

octa = Object(OCTAHEDRON, [x+300, y+300])
octa.rotate_points('x', 20)
octa.rotate_points('y', 20)

pygame.display.flip()

# ------------------------  Variables  ---------------------
rotation = None
move = None
curr_object = None
drag = False
change_pov = False
pov_drag = []

# Game loop
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_DOWN, pygame.K_UP, pygame.K_RIGHT, pygame.K_LEFT]:
                direction = pygame.key.name(event.key)
                if curr_object is not None:
                    rotation = direction
                else:
                    move = direction
            if event.key == pygame.K_t:
                if curr_object is not None:
                    rotation = 'twist'
                else:
                    move = 'twist'
            if event.key == pygame.K_TAB:
                change_draw_kwargs(curr_object, 'shadow')
            if event.key == pygame.K_1:
                change_draw_kwargs(curr_object, 'show_surfaces')
            if event.key == pygame.K_2:
                change_draw_kwargs(curr_object, 'show_nodes')
            if event.key == pygame.K_ESCAPE:
                run = False
            if event.key == pygame.K_h:
                print(env.space_pos)
            if event.key == pygame.K_SPACE:
                change_pov = not change_pov
        if event.type == pygame.KEYUP:
            rotation = None
            move = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            curr_object = object_choice((x, y))
            if curr_object is not None:
                drag = True
                x_diff, y_diff = x - curr_object.pos[0], y - curr_object.pos[1]
            else:
                if change_pov is True:
                    pov_drag.append((x, y))

        if event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            if drag is True:
                vector = np.array([x-x_diff-curr_object.pos[0], y-y_diff-curr_object.pos[1], 0])
                env.space_pos[curr_object.num] += vector
            elif pov_drag:
                prev_x, prev_y = pov_drag.pop()
                diff_x = x - prev_x
                diff_y = y - prev_y
                diff = max(diff_x, diff_y, key=lambda x: abs(x))
                sgn = 1 if diff > 0 else -1
                if abs(diff_x) > abs(diff_y):
                    direction = 'horizontal'
                else:
                    direction = 'vertical'

                env.rotate(direction, sgn*2)
                pov_drag.append((x, y))

                
        if event.type == pygame.MOUSEBUTTONUP:
            drag = False
            pov_drag = []

    screen.fill(bg_colour)

    if move is not None:
        env.move(move)

    rotation_values = rotations.get(rotation)
    if curr_object is not None:
        commands = commands1
        if rotation_values is not None:
            curr_object.rotate_points(rotation_values[0], rotation_values[1])
        curr_object.axis.draw(pos=(WIDTH-210, HEIGHT-30))
    else:
        commands = commands2
    
    env.process()
    draw_indicators(curr_object)
    show_commands(commands)

    pygame.display.flip()
    clock.tick(FPS)

pygame.QUIT
sys.exit()