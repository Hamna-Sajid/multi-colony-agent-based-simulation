import pygame
import random
import math
import numpy as np
import pandas as pd  # Import pandas

# --- CONFIG ---
SCREEN_SIZE = 600
GRID_DIMS = 50
CELL_SIZE = SCREEN_SIZE // GRID_DIMS
FPS = 10
NUM_COLONIES = 3
NUM_INITIAL_INDIVIDUALS = 5
REPRODUCTION_THRESHOLD = 100
REPRODUCTION_COST = 50

# --- Q-LEARNING CONFIG ---
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2
NUM_TRAINING_EPISODES = 200
TRAINING_STEPS_PER_EPISODE = 15

# --- TUNABLE PARAMETERS (Simulation) ---
AREA_WEIGHT = 500
RESOURCE_WEIGHT = 1.0
DISTANCE_WEIGHT = 0.5
CLOSE_DISTANCE_REWARD = 10
FREE_CELL_REWARD = 10
OWN_CELL_REWARD = 5
ENEMY_CELL_REWARD = 10
ENEMY_CELL_PENALTY = -3
FIGHT_WIN_REWARD = 50
FIGHT_LOSS_PENALTY = -3
GROUP_COHESION_REWARD = 1
BORDER_SPREAD_REWARD = 0

# --- INIT ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()

# --- COLORS ---
BASE_COLORS = [tuple(np.random.randint(0, 256, size=3)) for _ in range(NUM_COLONIES)]
LIGHT_COLORS = [(int(r*0.8), int(g*0.8), int(b*0.8)) for r, g, b in BASE_COLORS]

# --- GRID CELL ---
class Cell:
    """
    Represents a single cell in the simulation grid.

    Parameters
    ----------
    x : int, optional
        The x-coordinate of the cell (default is 0).
    y : int, optional
        The y-coordinate of the cell (default is 0).

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
    def __init__(self, x=0, y=0):
        self.colony_id = None
        self.individual = None
        self.x = x
        self.y = y

# --- INDIVIDUAL ---
class Individual:
    """
    Represents an individual agent belonging to a colony, with Q-learning capabilities.

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
    q_table : dict
        The Q-table for state-action values.
    """
    def __init__(self, x, y, colony):
        self.x, self.y = x, y
        self.colony = colony
        self.resources = 0
        self.q_table = {}

    def get_state(self, grid):
        """
        Get the current state representation for the individual based on its local grid.

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.

        Returns
        -------
        tuple
            Encoded state as a tuple of integers representing the local environment.
        """
        state = tuple()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
                    cell = grid[ny][nx]
                    if cell.individual:
                        if cell.individual.colony == self.colony:
                            state += (1,)  # Own individual
                        else:
                            state += (2,)  # Enemy individual
                    elif cell.colony_id is not None:
                        if cell.colony_id == self.colony.id:
                            state += (3,)  # Own territory
                        else:
                            state += (4,)  # Enemy territory
                    else:
                        state += (0,)  # Empty cell
                else:
                    state += (-1,) # Out of bounds
        return state

    def get_possible_actions(self):
        """
        Get all possible movement actions (including staying in place).

        Returns
        -------
        list of tuple of int
            List of (dx, dy) movement actions.
        """
        return [(0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1), (0, 0)] # Include staying

    def choose_action(self, grid, explore_rate):
        """
        Choose an action using an epsilon-greedy policy based on the Q-table.

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.
        explore_rate : float
            Probability of choosing a random action (exploration).

        Returns
        -------
        tuple
            The chosen (dx, dy) action.
        """
        state = self.get_state(grid)
        actions = self.get_possible_actions()
        if random.random() < explore_rate:
            return random.choice(actions)
        else:
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in actions}
            best_action = None
            max_q = -float('inf')
            random.shuffle(actions) # To break ties randomly
            for action in actions:
                if state in self.q_table and action in self.q_table[state]: #check if the action exists.
                    if self.q_table[state][action] > max_q:
                        max_q = self.q_table[state][action]
                        best_action = action
                else:
                    best_action = random.choice(actions)
            return best_action

    def evaluate_state_action(self, x, y, grid):
        """
        Evaluate the reward for moving to a given cell (state-action pair).

        Parameters
        ----------
        x : int
            The x-coordinate of the cell to evaluate.
        y : int
            The y-coordinate of the cell to evaluate.
        grid : list of list of Cell
            The simulation grid.

        Returns
        -------
        float
            The computed reward for the state-action pair.
        """
        cell = grid[y][x]
        reward = 0
        if cell.individual:
            if cell.individual.colony == self.colony:
                reward += CLOSE_DISTANCE_REWARD
            else:
                enemy = cell.individual
                my_score = self.colony.fitness() + self.resources
                enemy_score = enemy.colony.fitness() + enemy.resources
                reward += FIGHT_WIN_REWARD + enemy.resources if my_score > enemy_score else FIGHT_LOSS_PENALTY - enemy.resources
        elif cell.colony_id is None:
            reward += FREE_CELL_REWARD
        elif cell.colony_id == self.colony.id:
            reward += OWN_CELL_REWARD
        else:
            my_score = self.colony.fitness()
            enemy_score = colonies[cell.colony_id].fitness()
            reward += ENEMY_CELL_REWARD if my_score > enemy_score else ENEMY_CELL_PENALTY

        # Cohesion: Encourage moving towards colony center of mass
        if len(self.colony.individuals) > 1:
            cx = sum(i.x for i in self.colony.individuals) / len(self.colony.individuals)
            cy = sum(i.y for i in self.colony.individuals) / len(self.colony.individuals)
            dist_to_center = math.dist((x, y), (cx, cy))
            current_dist = math.dist((self.x, self.y), (cx, cy))
            if dist_to_center < current_dist:
                reward += GROUP_COHESION_REWARD

        # Exploration: Encourage spreading to borders with other colonies
        dirs = [(0,1), (1,0), (-1,0), (0,-1)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
                neighbor = grid[ny][nx]
                if neighbor.colony_id != self.colony.id and neighbor.colony_id is not None:
                    reward += BORDER_SPREAD_REWARD

        return reward

    def move(self, grid, action):
        """
        Move the individual according to the specified action.

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.
        action : tuple of int
            The (dx, dy) movement action.
        """
        dx, dy = action
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
            self.try_move_to(nx, ny, grid)

    def try_move_to(self, x, y, grid):
        """
        Attempt to move the individual to the specified cell, handling combat and territory capture.

        Parameters
        ----------
        x : int
            The x-coordinate of the target cell.
        y : int
            The y-coordinate of the target cell.
        grid : list of list of Cell
            The simulation grid.
        """
        target = grid[y][x]
        if target.individual:
            if target.individual.colony != self.colony:
                enemy = target.individual
                my_score = self.colony.fitness() + self.resources
                enemy_score = enemy.colony.fitness() + enemy.resources
                if my_score > enemy_score:
                    self.resources += enemy.resources
                    self.colony.resources += enemy.resources
                    enemy.colony.resources -= enemy.resources
                    # Check if the enemy is still in the list before removing
                    if enemy in enemy.colony.individuals:
                        enemy.colony.individuals.remove(enemy)
                    target.individual = self
                    grid[self.y][self.x].individual = None
                    self.x, self.y = x, y
                else:
                    self.colony.resources -= self.resources
                    target.individual.resources += self.resources
                    if self in self.colony.individuals:
                        self.colony.individuals.remove(self)
                    grid[self.y][self.x].individual = None
        else:
            if target.colony_id is None:
                self.resources += 10
                self.colony.resources += 10
                target.colony_id = self.colony.id
            elif target.colony_id != self.colony.id:
                if self.colony.fitness() > colonies[target.colony_id].fitness():
                    self.resources += 10
                    self.colony.resources += 10
                    colonies[target.colony_id].resources -= 10
                    target.colony_id = self.colony.id
            grid[self.y][self.x].individual = None
            target.individual = self
            self.x, self.y = x, y

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
    history : dict
        Historical data for population, area, and resources.
    """
    def __init__(self, id, center):
        self.id = id
        self.color = BASE_COLORS[id]
        self.light_color = LIGHT_COLORS[id]
        self.individuals = []
        self.center = center
        self.resources = 0
        self.history = {
            "population": [NUM_INITIAL_INDIVIDUALS],
            "area": [1],
            "resources": [0],
        }

    def spawn_individual(self, x, y):
        """
        Spawn a new individual at the specified location.

        Parameters
        ----------
        x : int
            The x-coordinate for the new individual.
        y : int
            The y-coordinate for the new individual.
        """
        ind = Individual(x, y, self)
        self.individuals.append(ind)
        grid[y][x].individual = ind
        grid[y][x].colony_id = self.id

    def fitness(self):
        """
        Compute the fitness of the colony based on area, resources, and average distance between individuals.

        Returns
        -------
        float
            The computed fitness value.
        """
        area = sum(1 for row in grid for cell in row if cell.colony_id == self.id)
        dist_sum = 0
        num_individuals = len(self.individuals)
        if num_individuals > 1:
            for i in range(num_individuals):
                for j in range(i + 1, num_individuals):
                    a, b = self.individuals[i], self.individuals[j]
                    dist_sum += math.dist((a.x, a.y), (b.x, b.y))
            avg_dist = dist_sum / (num_individuals * (num_individuals - 1) / 2)
        else:
            avg_dist = 0
        return AREA_WEIGHT * area + RESOURCE_WEIGHT * self.resources - DISTANCE_WEIGHT * avg_dist

    def reproduce(self):
        """
        Attempt to reproduce by spawning a new individual if resources allow.
        """
        if self.resources >= REPRODUCTION_THRESHOLD:
            self_grids = [cell for row in grid for cell in row if cell.colony_id == self.id]
            if self_grids:
                cell = random.choice(self_grids)
                if cell.individual is None:
                    self.spawn_individual(cell.x, cell.y)
                    self.resources -= REPRODUCTION_COST

    def update_history(self, grid):
        """
        Update the historical data for the colony (population, area, resources).

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.
        """
        area = sum(1 for row in grid for cell in row if cell.colony_id == self.id)
        self.history["population"].append(len(self.individuals))
        self.history["area"].append(area)
        self.history["resources"].append(self.resources)

# --- TRAINING PHASE ---
grid = [[Cell(x, y) for x in range(GRID_DIMS)] for y in range(GRID_DIMS)]
colonies = []
for i in range(NUM_COLONIES):
    cx, cy = random.randint(0, GRID_DIMS-1), random.randint(0, GRID_DIMS-1)
    colony = Colony(i, (cx, cy))
    colony.spawn_individual(cx, cy)
    for _ in range(NUM_INITIAL_INDIVIDUALS - 1):
        dirs = [(0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
        dx, dy = random.choice(dirs)
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS and grid[ny][nx].individual is None:
            colony.spawn_individual(nx, ny)
    colonies.append(colony)

print("Starting training...")
for episode in range(NUM_TRAINING_EPISODES):
    for step in range(TRAINING_STEPS_PER_EPISODE):
        all_individuals = [ind for col in colonies for ind in col.individuals]
        random.shuffle(all_individuals)
        for individual in all_individuals:
            old_state = individual.get_state(grid)
            action = individual.choose_action(grid, EXPLORATION_RATE)
            individual.move(grid, action)
            new_state = individual.get_state(grid)
            reward = individual.evaluate_state_action(individual.x, individual.y, grid)

            if old_state not in individual.q_table:
                individual.q_table[old_state] = {a: 0 for a in individual.get_possible_actions()}
            if action not in individual.q_table[old_state]:
                individual.q_table[old_state][action] = 0
            max_next_q = 0
            if new_state in individual.q_table:
                max_next_q = max(individual.q_table[new_state].values())
            
            if old_state in individual.q_table and action in individual.q_table[old_state]:
                individual.q_table[old_state][action] = individual.q_table[old_state][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - individual.q_table[old_state][action])
            else:
                individual.q_table[old_state][action] =  LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q)

    for colony in colonies:
        colony.reproduce()

    if episode % 100 == 0:
        print(f"Episode {episode} done.")

print("Training finished. Starting simulation.")

# --- SIMULATION PHASE ---
grid = [[Cell(x, y) for x in range(GRID_DIMS)] for y in range(GRID_DIMS)]
colonies = []
for i in range(NUM_COLONIES):
    cx, cy = random.randint(0, GRID_DIMS-1), random.randint(0, GRID_DIMS-1)
    colony = Colony(i, (cx, cy))
    colony.spawn_individual(cx, cy)
    for _ in range(NUM_INITIAL_INDIVIDUALS - 1):
        dirs = [(0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
        dx, dy = random.choice(dirs)
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS and grid[ny][nx].individual is None:
            colony.spawn_individual(nx, ny)
    colonies.append(colony)

running = True
runs = 0
while running and runs < 250:
    screen.fill((30,30,30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                x, y = event.pos
                x //= CELL_SIZE
                y //= CELL_SIZE
                if grid[y][x].individual is None:
                    BASE_COLORS.append(tuple(np.random.randint(0, 256, size=3)))
                    LIGHT_COLORS.append(tuple(int(c*0.8) for c in BASE_COLORS[-1]))
                    colony = Colony(len(colonies), (x, y))
                    colony.spawn_individual(x, y)
                    colonies.append(colony)
                    grid[y][x].colony_id = colony.id
                    grid[y][x].individual = colony.individuals[0]
            if event.button == 3:
                x, y = event.pos
                x //= CELL_SIZE
                y //= CELL_SIZE
                circle_radius = random.randint(1, 5)
                for dx in range(-circle_radius, circle_radius + 1):
                    for dy in range(-circle_radius, circle_radius + 1):
                        if dx ** 2 + dy ** 2 <= circle_radius ** 2:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
                                cell = grid[ny][nx]
                                if cell.individual:
                                    cell.individual.colony.individuals.remove(cell.individual)
                                    cell.individual = None
                                if cell.colony_id is not None:
                                    colonies[cell.colony_id].resources -= 10
                                    cell.colony_id = None

    for colony in colonies:
        for ind in colony.individuals[:]:
            action = ind.choose_action(grid, 0) # No exploration during simulation
            ind.move(grid, action)
        colony.reproduce()
        colony.update_history(grid) #update the history of the colonies

    for y in range(GRID_DIMS):
        for x in range(GRID_DIMS):
            cell = grid[y][x]
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell.colony_id is not None:
                pygame.draw.rect(screen, LIGHT_COLORS[cell.colony_id], rect)
            if cell.individual:
                pygame.draw.rect(screen, BASE_COLORS[cell.individual.colony.id], rect)

    for colony in colonies:
        if len(colony.individuals) > 0:
            cx = sum(ind.x for ind in colony.individuals) // len(colony.individuals)
            cy = sum(ind.y for ind in colony.individuals) // len(colony.individuals)
            cx_screen = cx * CELL_SIZE + CELL_SIZE // 2
            cy_screen = cy * CELL_SIZE + CELL_SIZE // 2
            for ind in colony.individuals:
                ind_x = ind.x *CELL_SIZE + CELL_SIZE // 2
                ind_y = ind.y *CELL_SIZE + CELL_SIZE // 2
                pygame.draw.line(screen, colony.color, (cx_screen, cy_screen), (ind_x, ind_y), 1)

    # Print colony resources
    for i, colony in enumerate(colonies):
        text = f"Colony {i}: {colony.resources} resources"
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10 + i * 20))

    pygame.display.flip()
    clock.tick(FPS)
    runs += 1

pygame.quit()

# --- Save data to CSV ---
all_colony_data = []  # List to store data for all colonies

for colony in colonies:
    history_data = colony.history
    colony_id = [colony.id] * len(history_data["population"])  # Create a list of colony IDs
    step_data = list(range(1, len(history_data["population"]) + 1)) #create a list of step counts
    colony_data = {
        "step": step_data, # Add step data
        "colony_id": colony_id,
        "population": history_data["population"],
        "area": history_data["area"],
        "resources": history_data["resources"]
    }
    all_colony_data.append(pd.DataFrame(colony_data))

# Concatenate all dataframes into one
final_df = pd.concat(all_colony_data, ignore_index=True)

# Save to CSV
final_df.to_csv("all_colonies_history.csv", index=False)
print("All colonies data saved to all_colonies_history.csv")
