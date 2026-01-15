import pygame
import random
import math
import os
import csv

# --- CONFIG ---
SCREEN_SIZE = 600
GRID_DIMS = 50
CELL_SIZE = SCREEN_SIZE // GRID_DIMS
FPS = 10
NUM_COLONIES = 3
NUM_INITIAL_INDIVIDUALS = 5
REPRODUCTION_THRESHOLD = 100
REPRODUCTION_COST = 50

# --- PSO CONFIG ---
SWARM_SIZE = 30
ITERATIONS = 250
INERTIA = 0.7
COGNITIVE_CONST = 1.4
SOCIAL_CONST = 1.4

# --- TUNABLE PARAMETERS (Simulation) ---
AREA_WEIGHT = 20
RESOURCE_WEIGHT = 1.0
DISTANCE_WEIGHT = 10
CLOSE_DISTANCE_REWARD = 10
FREE_CELL_REWARD = 10
OWN_CELL_REWARD = -30
ENEMY_CELL_REWARD = 10
ENEMY_CELL_PENALTY = -3
FIGHT_WIN_REWARD = 50
FIGHT_LOSS_PENALTY = -3
GROUP_COHESION_REWARD = 1
BORDER_SPREAD_REWARD = 3
MAX_STEPS = 250  # maximum number of simulation frames

# --- GRID CELL ---
class Cell:
    """
    Represents a single cell in the simulation grid.

    Parameters
    ----------
    x : int
        The x-coordinate of the cell.
    y : int
        The y-coordinate of the cell.

    Attributes
    ----------
    colony_id : int or None
        The ID of the colony that owns this cell, or None if unclaimed.
    individual : Individual or None
        The individual occupying this cell, if any.
    x : int
        The x-coordinate of the cell.
    y : int
        The y-coordinate of the cell.
    """
    def __init__(self, x, y):
        self.colony_id = None  # Colony that owns this cell
        self.individual = None  # Individual occupying this cell
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate

    def reset(self):
        """
        Reset the cell to its initial unclaimed and unoccupied state.
        """
        self.colony_id = None  # Remove colony ownership
        self.individual = None  # Remove individual

    def __repr__(self):
        """
        Return a string representation of the cell.

        Returns
        -------
        str
            String representation of the cell.
        """
        return f"Cell({self.x}, {self.y}, Colony: {self.colony_id}, Ind: {self.individual})"

# --- INDIVIDUAL ---
class Individual:
    """
    Represents an individual agent belonging to a colony.

    Parameters
    ----------
    x : int
        The x-coordinate of the individual's position.
    y : int
        The y-coordinate of the individual's position.
    colony : Colony
        The colony to which this individual belongs.

    Attributes
    ----------
    x : int
        The x-coordinate of the individual's position.
    y : int
        The y-coordinate of the individual's position.
    colony : Colony
        The colony to which this individual belongs.
    resources : int
        The resources currently held by the individual.
    fitness_value : float
        The fitness value of the individual.
    velocity : list of float
        The velocity vector for PSO updates.
    """
    def __init__(self, x, y, colony):
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate
        self.colony = colony  # Reference to parent colony
        self.resources = 0  # Resources held by this individual
        self.fitness_value = 0  # Fitness value for PSO
        self.velocity = [0, 0]  # Velocity vector for PSO

    def get_possible_actions(self):
        """
        Get all possible movement actions (including staying in place).

        Returns
        -------
        list of tuple of int
            List of (dx, dy) movement actions.
        """
        return [
            (0, 1),    # Down
            (1, 0),    # Right
            (-1, 0),   # Left
            (0, -1),   # Up
            (1, 1),    # Down-Right
            (-1, -1),  # Up-Left
            (1, -1),   # Up-Right
            (-1, 1),   # Down-Left
            (0, 0),    # Stay
        ]

    def move(self, grid, action):
        """
        Move the individual according to the specified action.

        Parameters
        ----------
        grid : Grid
            The simulation grid.
        action : tuple of int
            The (dx, dy) movement action.
        """
        dx, dy = action  # Unpack movement
        nx, ny = self.x + dx, self.y + dy  # New coordinates
        if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
            self.try_move_to(nx, ny, grid)  # Attempt move

    def try_move_to(self, x, y, grid):
        """
        Attempt to move the individual to the specified cell, handling combat and territory capture.

        Parameters
        ----------
        x : int
            The x-coordinate of the target cell.
        y : int
            The y-coordinate of the target cell.
        grid : Grid
            The simulation grid.
        """
        target = grid.cells[y][x]  # Target cell
        if target.individual:
            if target.individual.colony != self.colony:  # Enemy present
                enemy = target.individual
                my_score = self.colony.fitness(grid) + self.resources  # My strength
                enemy_score = enemy.colony.fitness(grid) + enemy.resources  # Enemy strength
                if my_score > enemy_score:
                    self.resources += enemy.resources  # Gain enemy's resources
                    self.colony.resources += enemy.resources  # Colony gains resources
                    enemy.colony.resources -= enemy.resources  # Enemy colony loses resources
                    if enemy in enemy.colony.individuals:
                        enemy.colony.individuals.remove(enemy)  # Remove enemy
                    target.individual = self  # Take over cell
                    grid.cells[self.y][self.x].individual = None  # Vacate old cell
                    self.x, self.y = x, y  # Update position
                else:
                    self.colony.resources -= self.resources  # Lose resources
                    target.individual.resources += self.resources  # Enemy gains resources
                    if self in self.colony.individuals:
                        self.colony.individuals.remove(self)  # Remove self
                    grid.cells[self.y][self.x].individual = None  # Vacate old cell
        else:
            if target.colony_id is None:
                self.resources += 10  # Gain for new territory
                self.colony.resources += 10
                target.colony_id = self.colony.id  # Claim cell
            elif target.colony_id != self.colony.id:
                if self.colony.fitness(grid) > grid.colonies[target.colony_id].fitness(grid):
                    self.resources += 10  # Gain for conquering
                    self.colony.resources += 10
                    grid.colonies[target.colony_id].resources -= 10  # Enemy loses resources
                    target.colony_id = self.colony.id  # Claim cell
            grid.cells[self.y][self.x].individual = None  # Vacate old cell
            target.individual = self  # Move to new cell
            self.x, self.y = x, y  # Update position

    def __repr__(self):
        """
        Return a string representation of the individual.

        Returns
        -------
        str
            String representation of the individual.
        """
        return f"Individual({self.x}, {self.y}, Colony: {self.colony.id})"

# --- COLONY ---
class Colony:
    """
    Represents a colony in the simulation.

    Parameters
    ----------
    id : int
        The unique identifier for the colony.
    center : tuple of int
        The (x, y) coordinates of the colony's initial center.
    color : tuple of int
        The RGB color for the colony's individuals.

    Attributes
    ----------
    id : int
        The unique identifier for the colony.
    color : tuple of int
        The RGB color for the colony's individuals.
    light_color : tuple of int
        The RGB color for the colony's territory.
    individuals : list of Individual
        The list of individuals belonging to the colony.
    center : tuple of int
        The (x, y) coordinates of the colony's initial center.
    resources : int
        The total resources accumulated by the colony.
    num_cells : int
        The number of cells owned by the colony.
    best_fitness : float
        The best fitness value found by the colony (for PSO).
    best_position : tuple of int
        The best position found by the colony (for PSO).
    """
    def __init__(self, id, center, color):
        self.id = id  # Colony ID
        self.color = color  # RGB color
        self.light_color = tuple(int(c * 0.8) for c in color)  # Lighter color for territory
        self.individuals = []  # List of individuals
        self.center = center  # Initial center
        self.resources = 0  # Total resources
        self.num_cells = 1  # Number of owned cells
        self.best_fitness = -float('inf')  # Best fitness for PSO
        self.best_position = center  # Best position for PSO

    def spawn_individual(self, x, y, grid):
        """
        Spawn a new individual at the specified location.

        Parameters
        ----------
        x : int
            The x-coordinate for the new individual.
        y : int
            The y-coordinate for the new individual.
        grid : Grid
            The simulation grid.
        """
        ind = Individual(x, y, self)  # Create new individual
        self.individuals.append(ind)  # Add to colony
        grid.cells[y][x].individual = ind  # Place on grid
        grid.cells[y][x].colony_id = self.id  # Mark cell as owned

    def fitness(self, grid):
        """
        Compute the fitness of the colony based on area, resources, and average distance between individuals.

        Parameters
        ----------
        grid : Grid
            The simulation grid.

        Returns
        -------
        float
            The computed fitness value.
        """
        area = sum(1 for row in grid.cells for cell in row if cell.colony_id == self.id)  # Area owned
        dist_sum = 0
        num_individuals = len(self.individuals)
        if num_individuals > 1:
            for i in range(num_individuals):
                for j in range(i + 1, num_individuals):
                    a, b = self.individuals[i], self.individuals[j]
                    dist_sum += math.dist((a.x, a.y), (b.x, b.y))  # Pairwise distances
            avg_dist = dist_sum / (num_individuals * (num_individuals - 1) / 2)  # Average distance
        else:
            avg_dist = 0
        return AREA_WEIGHT * area + RESOURCE_WEIGHT * self.resources - DISTANCE_WEIGHT * avg_dist  # Fitness formula

    def reproduce(self, grid):
        """
        Attempt to reproduce by spawning a new individual if resources allow.

        Parameters
        ----------
        grid : Grid
            The simulation grid.
        """
        if self.resources >= REPRODUCTION_THRESHOLD:
            self_grids = [
                cell for row in grid.cells for cell in row if cell.colony_id == self.id
            ]
            if self_grids:
                cell = random.choice(self_grids)  # Pick random owned cell
                if cell.individual is None:
                    self.spawn_individual(cell.x, cell.y, grid)  # Spawn new individual
                    self.resources -= REPRODUCTION_COST  # Pay cost

    def update_num_cells(self, grid):
        """
        Update the number of cells owned by the colony.

        Parameters
        ----------
        grid : Grid
            The simulation grid.
        """
        self.num_cells = sum(
            1 for row in grid.cells for cell in row if cell.colony_id == self.id
        )  # Update owned cell count

    def __repr__(self):
        """
        Return a string representation of the colony.

        Returns
        -------
        str
            String representation of the colony.
        """
        return f"Colony({self.id}, Center: {self.center}, Color: {self.color})"

# --- GRID ---
class Grid:
    """
    Represents the simulation grid containing cells and colonies.

    Parameters
    ----------
    dims : int
        The dimensions (width and height) of the grid.

    Attributes
    ----------
    dims : int
        The dimensions of the grid.
    cells : list of list of Cell
        The 2D array of cells in the grid.
    colonies : list of Colony
        The list of colonies present in the grid.
    """
    def __init__(self, dims):
        self.dims = dims  # Grid size
        self.cells = [[Cell(x, y) for x in range(dims)] for y in range(dims)]  # 2D grid
        self.colonies = []  # List of colonies

    def reset(self):
        """
        Reset all cells and colonies in the grid.
        """
        for row in self.cells:
            for cell in row:
                cell.reset()  # Reset each cell
        self.colonies = []  # Remove all colonies

    def initialize_colonies(self, num_colonies, num_initial_individuals, unique_colors):
        """
        Initialize colonies at random positions with initial individuals.

        Parameters
        ----------
        num_colonies : int
            Number of colonies to initialize.
        num_initial_individuals : int
            Number of initial individuals per colony.
        unique_colors : list of tuple
            List of unique RGB color tuples for each colony.
        """
        self.colonies = []
        for i in range(num_colonies):
            cx, cy = random.randint(0, self.dims - 1), random.randint(
                0, self.dims - 1
            )  # Random center
            colony = Colony(i, (cx, cy), unique_colors[i])
            colony.spawn_individual(cx, cy, self)
            for _ in range(num_initial_individuals - 1):
                dirs = [
                    (0, 1),
                    (1, 0),
                    (-1, 0),
                    (0, -1),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]
                dx, dy = random.choice(dirs)
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < self.dims
                    and 0 <= ny < self.dims
                    and self.cells[ny][nx].individual is None
                ):
                    colony.spawn_individual(nx, ny, self)
            self.colonies.append(colony)

    def draw(self, screen):
        """
        Draw the grid, colonies, and individuals on the Pygame screen.

        Parameters
        ----------
        screen : pygame.Surface
            The Pygame surface to draw on.
        """
        for y in range(self.dims):
            for x in range(self.dims):
                cell = self.cells[y][x]
                rect = pygame.Rect(
                    x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE
                )
                if cell.colony_id is not None:
                    pygame.draw.rect(screen, self.colonies[cell.colony_id].light_color, rect)
                if cell.individual:
                    pygame.draw.rect(
                        screen, self.colonies[cell.individual.colony.id].color, rect
                    )

        for colony in self.colonies:
            if len(colony.individuals) > 0:
                cx = sum(ind.x for ind in colony.individuals) // len(
                    colony.individuals
                )
                cy = sum(ind.y for ind in colony.individuals) // len(
                    colony.individuals
                )
                cx_screen = cx * CELL_SIZE + CELL_SIZE // 2
                cy_screen = cy * CELL_SIZE + CELL_SIZE // 2
                for ind in colony.individuals:
                    ind_x = ind.x * CELL_SIZE + CELL_SIZE // 2
                    ind_y = ind.y * CELL_SIZE + CELL_SIZE // 2
                    pygame.draw.line(
                        screen,
                        colony.color,
                        (cx_screen, cy_screen),
                        (ind_x, ind_y),
                        1,
                    )

    def __repr__(self):
        """
        Return a string representation of the grid.

        Returns
        -------
        str
            String representation of the grid.
        """
        return f"Grid({self.dims}, Colonies: {len(self.colonies)})"

# --- FUNCTIONS ---
def generate_unique_colors(num_colors):
    """
    Generate a list of unique RGB colors.

    Parameters
    ----------
    num_colors : int
        The number of unique colors to generate.

    Returns
    -------
    list of tuple
        List of unique RGB color tuples.
    """
    colors = []
    while len(colors) < num_colors:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = (r, g, b)
        if color not in colors:
            colors.append(color)
    return colors

def calculate_fitness(individual, grid):
    """
    Calculate the fitness of an individual based on its position and environment.

    Parameters
    ----------
    individual : Individual
        The individual whose fitness is to be calculated.
    grid : Grid
        The simulation grid.

    Returns
    -------
    float
        The computed fitness value.
    """
    x, y = individual.x, individual.y
    cell = grid.cells[y][x]
    reward = 0
    if cell.individual:
        if cell.individual.colony == individual.colony:
            reward += CLOSE_DISTANCE_REWARD
        else:
            enemy = cell.individual
            my_score = individual.colony.fitness(grid) + individual.resources
            enemy_score = enemy.colony.fitness(grid) + enemy.resources
            reward += (
                FIGHT_WIN_REWARD + enemy.resources
                if my_score > enemy_score
                else FIGHT_LOSS_PENALTY - enemy.resources
            )
    elif cell.colony_id is None:
        reward += FREE_CELL_REWARD
    elif cell.colony_id == individual.colony.id:
        reward += OWN_CELL_REWARD
    else:
        my_score = individual.colony.fitness(grid)
        enemy_score = grid.colonies[cell.colony_id].fitness(grid)
        reward += (
            ENEMY_CELL_REWARD if my_score > enemy_score else ENEMY_CELL_PENALTY
        )

    # Encourage individuals to move toward the centroid of their colony
    if len(individual.colony.individuals) > 1:
        cx = sum(i.x for i in individual.colony.individuals) / len(
            individual.colony.individuals
        )
        cy = sum(i.y for i in individual.colony.individuals) / len(
            individual.colony.individuals
        )
        dist_to_center = math.dist((x, y), (cx, cy))
        current_dist = math.dist((individual.x, individual.y), (cx, cy))
        if dist_to_center < current_dist:
            reward += GROUP_COHESION_REWARD

    # Encourage exploring border areas
    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
            neighbor = grid.cells[ny][nx]
            if (
                neighbor.colony_id != individual.colony.id
                and neighbor.colony_id is not None
            ):
                reward += BORDER_SPREAD_REWARD
    return reward

def update_particle_velocity(individual, best_position, global_best_position):
    """
    Update the velocity of an individual (particle) in PSO.

    Parameters
    ----------
    individual : Individual
        The individual whose velocity is to be updated.
    best_position : tuple of int
        The best position found by the individual (personal best).
    global_best_position : tuple of int
        The best position found by the swarm (global best).
    """
    inertia = INERTIA
    cognitive_const = COGNITIVE_CONST
    social_const = SOCIAL_CONST

    for i in range(2):  # Update x and y components of velocity
        personal_best = best_position[i] - individual.x if i == 0 else best_position[i] - individual.y
        global_best = global_best_position[i] - individual.x if i == 0 else global_best_position[i]

        individual.velocity[i] = (
            inertia * individual.velocity[i]
            + cognitive_const * random.random() * personal_best
            + social_const * random.random() * global_best
        )

def clamp_position(x, lower_bound, upper_bound):
    """
    Clamp a position value within specified bounds.

    Parameters
    ----------
    x : int or float
        The value to clamp.
    lower_bound : int or float
        The lower bound.
    upper_bound : int or float
        The upper bound.

    Returns
    -------
    int or float
        The clamped value.
    """
    return max(lower_bound, min(x, upper_bound))

def main():
    """
    Main function to run the PSO-based colony simulation.
    """
    # --- INIT ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()

    # --- Simulation Setup ---
    grid = Grid(GRID_DIMS)
    unique_colors = generate_unique_colors(NUM_COLONIES)

    # --- Data Storage Setup (CSV) ---
    os.makedirs("simulation_data", exist_ok=True)
    csv_file_path = "simulation_data/simulation_data.csv"
    csv_headers = ["Time", "Colony", "Area", "Population", "Resources"]
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

    def reset_simulation():
        """
        Reset the simulation grid and initialize colonies.
        """
        grid.reset()
        grid.initialize_colonies(NUM_COLONIES, NUM_INITIAL_INDIVIDUALS, unique_colors)

    reset_simulation()

    # --- PSO Setup ---
    global_best_fitness = -float('inf')
    global_best_position = (0, 0)
    swarm = []
    for colony in grid.colonies:
        for ind in colony.individuals:
            swarm.append(ind)  # Flatten the individuals into a single list
    # --- SIMULATION PHASE ---
    simulation_time = 0
    running = True
    while running and simulation_time < MAX_STEPS:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    x //= CELL_SIZE
                    y //= CELL_SIZE
                    if grid.cells[y][x].individual is None:
                        unique_colors = generate_unique_colors(len(grid.colonies) + 1)
                        colony = Colony(
                            len(grid.colonies), (x, y), unique_colors[-1]
                        )
                        colony.spawn_individual(x, y, grid)
                        grid.colonies.append(colony)
                        grid.cells[y][x].colony_id = colony.id
                        grid.cells[y][x].individual = colony.individuals[0]
                if event.button == 3:
                    x, y = event.pos
                    x //= CELL_SIZE
                    y //= CELL_SIZE
                    circle_radius = random.randint(1, 5)
                    for dx in range(-circle_radius, circle_radius + 1):
                        for dy in range(-circle_radius, circle_radius + 1):
                            if dx**2 + dy**2 <= circle_radius**2:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
                                    cell = grid.cells[ny][nx]
                                    if cell.individual:
                                        if (
                                            cell.individual
                                            in cell.individual.colony.individuals
                                        ):
                                            cell.individual.colony.individuals.remove(
                                                cell.individual
                                            )
                                        cell.individual = None
                                    if cell.colony_id is not None:
                                        grid.colonies[cell.colony_id].resources -= 10
                                        cell.colony_id = None

        # --- PSO Main Loop ---
        for iteration in range(1): #  No need for more than 1 iteration, the individuals move at every timestep
            for individual in swarm:
                # Calculate fitness
                individual.fitness_value = calculate_fitness(individual, grid)

                # Update personal best
                if individual.fitness_value > individual.colony.best_fitness:
                    individual.colony.best_fitness = individual.fitness_value
                    individual.colony.best_position = (individual.x, individual.y)

                # Update global best
                if individual.colony.best_fitness > global_best_fitness:
                    global_best_fitness = individual.colony.best_fitness
                    global_best_position = individual.colony.best_position

            for individual in swarm:
                # Update velocity
                update_particle_velocity(individual, individual.colony.best_position, global_best_position)

                # Move particle
                new_x = int(round(individual.x + individual.velocity[0]))
                new_y = int(round(individual.y + individual.velocity[1]))

                # Clamp the new positions
                new_x = clamp_position(new_x, 0, GRID_DIMS - 1)
                new_y = clamp_position(new_y, 0, GRID_DIMS - 1)
                possible_actions = individual.get_possible_actions()
                # Find the action that gets us closest to the new_x and new_y
                best_action = (0,0)
                min_dist = float('inf')
                for action in possible_actions:
                    dx, dy = action
                    next_x, next_y = individual.x + dx, individual.y + dy
                    if 0 <= next_x < GRID_DIMS and 0 <= next_y < GRID_DIMS:
                       dist = math.dist((next_x,next_y),(new_x,new_y))
                       if dist < min_dist:
                           min_dist = dist
                           best_action = action
                individual.move(grid,best_action)

        for colony in grid.colonies:
            colony.reproduce(grid)
            colony.update_num_cells(grid)

        grid.draw(screen)

        for i, colony in enumerate(grid.colonies):
            text = f"Colony {i}: {max(colony.resources,0)} resources, Pop: {len(colony.individuals)}, Cells: {colony.num_cells}"
            font = pygame.font.Font(None, 24)
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(FPS)
        simulation_time += 1

        # --- Data Logging (CSV) ---
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            for colony in grid.colonies:
                writer.writerow(
                    [
                        simulation_time,
                        colony.id,
                        colony.num_cells,
                        len(colony.individuals),
                        colony.resources,
                    ]
                )

    pygame.quit()
    print(f"Simulation ended after {simulation_time} steps.")

if __name__ == "__main__":
    main()
