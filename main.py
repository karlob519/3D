# Imports
import pygame
import numpy as np
import numpy.linalg as LA
import sys
import backend

# Display
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = backend.WIDTH, backend.HEIGHT
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
a, b, c = 75, 50, 100
CUBOID = backend.Cuboid(a, b, c)
CUBE = backend.Cuboid(a, a, a)
PYRAMID = backend.Pyramid(a, b, c)
OCTAHEDRON = backend.Octahedron(a, b, c)
CUBE.name = 'cube'
CUBE.ind_coords = [[-30, -30], [-30, 30], [30, 30], [30, -30]]

origin = np.array([0, 0, 0])
env = backend.Environment(start_POV=np.array([0, 0, 500]))

light_source = np.array([WIDTH/2, 1000, 0])

x_vec = np.array([100, 0, 0])
y_vec = np.array([0, 100, 0])
z_vec = np.array([0, 0, 100])
xyz_vecs = [x_vec, y_vec, z_vec]

# Define command strings for user instructions.
left1, left2, left3 = f'{chr(8592)} : rotate left', f'{chr(8592)} : move left', f'{chr(8592)} : look left'
right1, right2, right3 = f'{chr(8594)} : rotate right', f'{chr(8594)} : move right', f'{chr(8594)} : look right'
up1, up2, up3 = f'{chr(8593)} : rotate up', f'{chr(8593)} : move up', f'{chr(8593)} : look up'
down1, down2, down3 = f'{chr(8595)} : rotate down', f'{chr(8595)} : move down', f'{chr(8595)} : look down'
forward = f'{chr(8593)} : move forward'
back = f'{chr(8595)} : move back'
t = f't : twist'
one = '1 : toggle surfaces'
two = '2 : toggle nodes'
tab = 'TAB : toggle rotate FOV'
space = 'SPACE : toggle ahead / sideways'

# Define lists of commands for different contexts.
commands1 = [left1, right1, up1, down1, t, one, two, tab, space]
commands2 = [left2, right2, up2, down2, one, two, tab, space]
commands3 = [left2, right2, forward, back, one, two, tab, space]
commands4 = [left3, right3, up3, down3, one, two, tab, space]


# ---------------------  Classes  ------------------------ 
class Object:
    def __init__(self, shape, position: np.ndarray, environment: backend.Environment = env):
        # Initialize an Object with a specified shape, position, and environment.
        self.shape = shape
        self.shape.env = environment
        self.pos = position
        self.nodes = set_object(shape.nodes, position)
        self.nodes['center'] = position
        self.axis = Axis(self.pos)
        self.num = len(env)  # Store the number of objects in the environment.
        self.draw_kwargs = {
            'edge_colour': black,   # Default edge color.
            'show_surfaces': True,  # Show surfaces by default.
            'show_nodes': False,    # Do not show nodes by default.
            'shadow': False,        # Do not show shadows by default.
            'axis': False,          # Do not show the axis by default.
            'show_edges': True      # Show edges by default.
        }
        self.env = environment
        environment.add(self)  # Add the object to the environment.

    def rotate_points(self, direction: str, angle: float = 5):
        # Rotate the object's points in a specified direction by a given angle.
        alpha = to_rad(angle)
        if direction == 'x':
            M = x_rotation(alpha)  # Create a rotation matrix for X-axis rotation.
        elif direction == 'y':
            M = y_rotation(alpha)  # Create a rotation matrix for Y-axis rotation.
        else:
            M = z_rotation(alpha)  # Create a rotation matrix for Z-axis rotation.

        for node, vector in self.nodes.items():
            self.nodes[node] = rotate_point(M, vector, self.pos)

        self.axis.rotate_points(direction, angle)  # Rotate the axis.

    def boundary(self):
        # Calculate and return the boundary points of the object.
        pygame_coords = self.env.find_2d_nodes(self.nodes)
        center = pygame_coords['center']
        points = sorted(pygame_coords, key=lambda x: LA.norm(pygame_coords[x] - center))
        points.remove('center')
        return points[-4:]

    def in_range(self, point: tuple) -> bool:
        # Check if a given point is within the object's boundary.
        pygame_coords = self.env.find_2d_nodes(self.nodes)
        boundary = self.boundary()

        p_vec = np.array(point)

        return point_in_shape(boundary, p_vec, pygame_coords)

    def draw(self, edge_colour=black, show_nodes: bool = False, show_surfaces: bool = False, shadow: bool = False, axis: bool = False, show_edges: bool = False):
        # Draw the object with specified display options.
        not_shown = self.shape.find_not_shown(self.nodes)
        pygame_coords = self.env.find_2d_nodes(self.nodes)

        if show_nodes is True:
            for node, vector in pygame_coords.items():
                if node != 'center' and node not in not_shown:
                    pygame.draw.circle(screen, red, vector, 10)  # Draw a red circle for each node.
                    str_val = node
                    val = node_font.render(str_val, True, light_blue)
                    screen.blit(val, vector)  # Display the node name.

        if axis is True:
            self.axis.draw()  # Draw the axis.

        if show_surfaces is True:
            self.shape.draw_surfaces(self.nodes, screen)  # Draw the object's surfaces.

        if show_edges is True:
            for pair in self.shape.edges:
                if pair in self.shape.show_edges(self.nodes):
                    p1, p2 = pair[0], pair[1]
                    node1, node2 = pygame_coords[p1], pygame_coords[p2]
                    if not_shown is not None and p1 not in not_shown and p2 not in not_shown:
                        draw_line(node1, node2, edge_colour)  # Draw edges between nodes.
                    elif not_shown is None:
                        draw_line(node1, node2, edge_colour)
                        

        return

    
class Axis:
    def __init__(self, center: np.ndarray):
        # Initialize an Axis object with a center point and default angles.
        self.center = center
        self.vectors = [x_vec, y_vec, z_vec]  # List of axis vectors (x, y, z).
        self.angles = [0, 0, 0]  # Default rotation angles for each axis.

    def rotate_points(self, direction: str, angle: float = 5):
        # Rotate the axis vectors based on a specified direction and angle.
        alpha = to_rad(angle)  # Convert angle to radians.
        if direction == 'x':
            M = x_rotation(alpha)  # Create a rotation matrix for X-axis rotation.
        elif direction == 'y':
            M = y_rotation(alpha)  # Create a rotation matrix for Y-axis rotation.
        else:
            M = z_rotation(alpha)  # Create a rotation matrix for Z-axis rotation.

        for i in range(3):
            vector = self.vectors[i]
            self.vectors[i] = rotate_point(M, vector, self.center)  # Rotate each axis vector.
            self.angles[i] = find_angle(self.vectors[i], xyz_vecs[i])  # Update rotation angles.

    def draw(self, colour=black, pos=(0, 0)):
        # Draw the current rotation angles for each axis.
        x, y, z = self.angles
        angles_str = f'X: {x}   Y: {y}   Z: {z}'  # Format angle values as a string.
        angles_val = ariel.render(angles_str, True, colour)  # Render the string with the specified color.
        screen.blit(angles_val, pos)  # Display the rendered angles string at the specified position.

            
# ------------------------  Functions  -------------------------
rotate_point = backend.rotate_point
x_rotation = backend.x_rotation
y_rotation = backend.y_rotation
z_rotation = backend.z_rotation
to_rad = backend.to_rad
to_deg = backend.to_deg
find_angle = backend.find_angle
to_pygame = backend.to_pygame_coords
set_object = backend.set_object
point_in_shape = backend.point_in_shape


def draw_line(start_point: tuple, end_point: tuple, colour: tuple, surface=screen, flip: bool=False):
    pygame.draw.line(surface, colour, start_point, end_point, width=2)
    if flip:
        pygame.display.flip()
    return


def draw_polygon(point_list: list, pos: tuple, colour: tuple, surface: pygame.Surface=screen, fill: bool=False):
    w = 0 if fill is True else 1
    vec = np.array(pos)
    points_to_draw = [np.array(point) + vec for point in point_list]
    pygame.draw.polygon(surface, colour, points_to_draw, width=w)

    return


def extend_line(point1:tuple, point2: tuple, delta_y: float) -> tuple:
    x1, y1 = point1
    x2, y2 = point2
    a = (y2-y1) / (x2-x1) if x2 != x1 else 1 
    x = (y2 + delta_y - y1) / a + x1
    return np.array([x, y2 + delta_y])


def object_choice(pos: tuple, env: backend.Environment=env) -> Object:
    for obj in env:
        if obj.in_range(pos) is True:
            return obj
    else:
        return None
    

def draw_indicators(curr_obj: Object):
    curr_num = curr_obj.num if curr_obj is not None else None
    n = len(env)
    for i in range(n):
        obj = env[i]
        pos = (i*75 + 40, HEIGHT-50)
        if i == curr_num:
            draw_polygon(obj.shape.ind_coords, pos, red, fill=True)
        else:
            draw_polygon(obj.shape.ind_coords, pos, black)
    
    return


def change_draw_kwargs(curr_obj: Object, key: str, env: backend.Environment=env):    
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


def show_POV(position: tuple=(WIDTH-400, 25), env: backend.Environment=env, colour: tuple=black):
    POV_vals = np.array([round(el, 2) for el in env.POV])
    text = f'POV:     {POV_vals}'
    val = calibri.render(text, True, colour)
    screen.blit(val, position)


def show_FOV_dir(position: tuple=(WIDTH-400, 50), env: backend.Environment=env, colour: tuple=black):
    vector = [round(el, 2) for el in env.FOV_vector]
    text = f'FOV vector: {np.array(vector)}'
    val = calibri.render(text, True, colour)
    screen.blit(val, position)


def draw_origin(colour: tuple=dark_blue, env: backend.Environment=env):
    pos = env.find_pygame_xy(origin)
    dist = LA.norm(env.POV)
    pygame.draw.circle(screen, colour, pos, 150/np.sqrt(dist))
    return


# ----------------------  Objects  ----------------------------
start_pos1 = np.array([0, 0, 200])
start_pos2 = np.array([200, 200, -200])
start_pos3 = np.array([-200, 200, -200])

# cuboid = Object(CUBOID, start_pos2)
# cube = Object(CUBE, start_pos1)
pyramid = Object(PYRAMID, start_pos2)
# octa = Object(OCTAHEDRON, start_pos3)


pygame.display.flip()

# ------------------------  Variables  ---------------------
rotation = None
move = None
curr_object = None
drag = False
change_pov = False
for_back = False
FOV_rotate = False
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
                    if for_back is True:
                        if event.key == pygame.K_UP:
                            direction = 'forward'
                        elif event.key == pygame.K_DOWN:
                            direction = 'back'
                    move = direction
            if event.key == pygame.K_t:
                if curr_object is not None:
                    rotation = 'twist'
                else:
                    move = 'twist'
            if event.key == pygame.K_TAB:
                FOV_rotate = not FOV_rotate
                for_back = False
            if event.key == pygame.K_1:
                change_draw_kwargs(curr_object, 'show_surfaces')
            if event.key == pygame.K_2:
                change_draw_kwargs(curr_object, 'show_nodes')
            if event.key == pygame.K_ESCAPE:
                run = False
            if event.key == pygame.K_h:
                if curr_object is not None:
                    print(curr_object.shape.find_not_shown(curr_object.nodes, debug=True))
            if event.key == pygame.K_SPACE:
                for_back = not for_back
        if event.type == pygame.KEYUP:
            rotation = None
            move = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            curr_object = object_choice((x, y))
            if curr_object is not None:
                drag = True
                center = env.find_pygame_xy(curr_object.pos)
                x_diff, y_diff = x - center[0], y - center[1]
            else:
                if change_pov is True:
                    pov_drag.append((x, y))

        if event.type == pygame.MOUSEMOTION:
            x, y = to_pygame(pygame.mouse.get_pos())
            if drag is True:
                None
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
        if FOV_rotate is False:
            env.translate(move)
        else:
            env.rotate_FOV(1/12, move)

    rotation_values = rotations.get(rotation)
    if curr_object is not None:
        commands = commands1
        if rotation_values is not None:
            curr_object.rotate_points(rotation_values[0], rotation_values[1])
        curr_object.axis.draw(pos=(WIDTH-210, HEIGHT-30))
    else:
        if FOV_rotate is True:
            commands = commands4
        else:
            if for_back is True:
                commands = commands3
            else:
                commands = commands2
    
    env.process()
    draw_indicators(curr_object)
    draw_origin(colour=black)
    show_commands(commands)
    show_POV()
    show_FOV_dir()

    pygame.display.flip()
    clock.tick(FPS)

pygame.QUIT
sys.exit()