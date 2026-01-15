import pygame
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os # For path joining
import csv # Import the csv module

# --- CONFIG ---
SCREEN_SIZE = 600
GRID_DIMS = 100
CELL_SIZE = SCREEN_SIZE // GRID_DIMS
FPS = 10  # Simulation frames per second
SIMULATION_DURATION_STEPS = 250 # Run simulation for a fixed number of steps for automated data collection
# Set to None to run indefinitely until window is closed. If None, plots appear after closing window.


NUM_COLONIES = 3
NUM_INITIAL_INDIVIDUALS = 5
REPRODUCTION_THRESHOLD = 100
REPRODUCTION_COST = 50

# --- TUNABLE PARAMETERS ---
AREA_WEIGHT = 10
RESOURCE_WEIGHT = 1.0
DISTANCE_WEIGHT = 0.0

ENEMY_CELL_PENALTY = 3
FIGHT_LOSS_PENALTY = -7

CLOSE_DISTANCE_REWARD = 3
FREE_CELL_REWARD = 10
FIGHT_WIN_REWARD = 10
OWN_CELL_REWARD = 0
ENEMY_CELL_REWARD = 10
GROUP_COHESION_REWARD = 0
BORDER_SPREAD_REWARD = 0

# --- INIT ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Colony Simulation")
clock = pygame.time.Clock()

# --- COLORS ---
# Initial colors for pre-defined colonies
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
    """
    def __init__(self, x, y, colony):
        self.x, self.y = x, y
        self.colony = colony
        self.resources = 0 # Individual resources, e.g., gathered before returning to colony or for combat strength

    def move(self, grid, colonies_list):
        """
        Move the individual to the best adjacent cell based on a greedy evaluation.

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.
        colonies_list : list of Colony
            List of all colonies in the simulation.
        """
        # Possible directions for movement, including diagonals
        dirs = [(0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
        random.shuffle(dirs) # Randomize direction choice
        best_cell_coords = (self.x, self.y) # Current position is default best
        best_score = -float('inf') # Initialize best score

        # Evaluate all adjacent cells
        for dx, dy in dirs:
            nx, ny = self.x + dx, self.y + dy
            # Check if the new position is within grid boundaries
            if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
                score = self.evaluate_cell(nx, ny, grid, colonies_list)
                if score > best_score:
                    best_score = score
                    best_cell_coords = (nx, ny)

        # If a better cell is found, attempt to move
        if best_cell_coords != (self.x, self.y):
            self.try_move_to(best_cell_coords[0], best_cell_coords[1], grid, colonies_list)

    def evaluate_cell(self, x, y, grid, colonies_list):
        """
        Evaluate the desirability of moving to a given cell.

        Parameters
        ----------
        x : int
            The x-coordinate of the cell to evaluate.
        y : int
            The y-coordinate of the cell to evaluate.
        grid : list of list of Cell
            The simulation grid.
        colonies_list : list of Colony
            List of all colonies in the simulation.

        Returns
        -------
        score : float
            The computed score for the cell.
        """
        cell = grid[y][x]
        score = 0
        
        # Evaluate based on cell occupant
        if cell.individual: # If cell is occupied by another individual
            if cell.individual.colony == self.colony: # Friendly individual
                score += CLOSE_DISTANCE_REWARD # Reward for being near friendlies
            else: # Enemy individual
                enemy = cell.individual
                if enemy.colony and enemy.colony.id < len(colonies_list): # Check if enemy colony is valid
                    my_score = self.colony.fitness(grid) + self.resources
                    enemy_colony_obj = colonies_list[enemy.colony.id]
                    enemy_colony_fitness = enemy_colony_obj.fitness(grid) if enemy_colony_obj else 0
                    enemy_score_val = enemy_colony_fitness + enemy.resources
                    # Score based on potential fight outcome
                    score += FIGHT_WIN_REWARD + enemy.resources if my_score > enemy_score_val else FIGHT_LOSS_PENALTY - enemy.resources
                else: # Invalid enemy colony
                    score += FIGHT_LOSS_PENALTY
        elif cell.colony_id is None: # Unclaimed cell
            score += FREE_CELL_REWARD
        elif cell.colony_id == self.colony.id: # Own colony's territory
            score += OWN_CELL_REWARD
        else: # Enemy colony's territory (but no individual)
            if cell.colony_id is not None and cell.colony_id < len(colonies_list): # Check if enemy colony is valid
                my_score = self.colony.fitness(grid)
                enemy_colony_obj = colonies_list[cell.colony_id]
                enemy_score_val = enemy_colony_obj.fitness(grid) if enemy_colony_obj else 0
                # Score based on potential territory capture
                score += ENEMY_CELL_REWARD if my_score > enemy_score_val else ENEMY_CELL_PENALTY
            else: # Invalid enemy colony
                score += ENEMY_CELL_PENALTY

        # Cohesion: Encourage moving towards colony center of mass
        if len(self.colony.individuals) > 1:
            # Calculate colony's center of mass (average position of individuals)
            cx = sum(i.x for i in self.colony.individuals) / len(self.colony.individuals)
            cy = sum(i.y for i in self.colony.individuals) / len(self.colony.individuals)
            dist_to_center = math.dist((x, y), (cx, cy))
            current_dist = math.dist((self.x, self.y), (cx, cy))
            if dist_to_center < current_dist: # If moving to (x,y) is closer to center
                score += GROUP_COHESION_REWARD

        # Exploration: Encourage spreading to borders with other colonies
        dirs_border = [(0,1), (1,0), (-1,0), (0,-1)] # Cardinal directions
        random.shuffle(dirs_border)
        for dx, dy in dirs_border:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS:
                neighbor = grid[ny][nx]
                # If neighbor cell is enemy territory
                if neighbor.colony_id != self.colony.id and neighbor.colony_id is not None:
                    score += BORDER_SPREAD_REWARD
        return score

    def try_move_to(self, x, y, grid, colonies_list):
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
        colonies_list : list of Colony
            List of all colonies in the simulation.
        """
        target_cell = grid[y][x]
        current_cell = grid[self.y][self.x]

        if target_cell.individual: # Cell occupied by another individual
            if target_cell.individual.colony != self.colony: # It's an enemy
                enemy = target_cell.individual
                if enemy.colony and enemy.colony.id < len(colonies_list) and colonies_list[enemy.colony.id]: # Valid enemy
                    my_score = self.colony.fitness(grid) + self.resources
                    enemy_colony_obj = colonies_list[enemy.colony.id]
                    enemy_colony_fitness = enemy_colony_obj.fitness(grid)
                    enemy_score_val = enemy_colony_fitness + enemy.resources

                    if my_score > enemy_score_val: # Win fight
                        self.resources += enemy.resources # Gain enemy's individual resources
                        self.colony.total_resources_collected += enemy.resources # Colony gains resources
                        
                        enemy_colony_obj.total_resources_collected = max(0, enemy_colony_obj.total_resources_collected - enemy.resources) # Enemy colony loses resources
                        
                        if enemy in enemy_colony_obj.individuals: # Remove enemy individual
                            enemy_colony_obj.individuals.remove(enemy)
                        
                        target_cell.individual = self # Move to target cell
                        current_cell.individual = None # Vacate current cell
                        self.x, self.y = x, y
                        target_cell.colony_id = self.colony.id # Claim cell for own colony
                    else: # Lose fight
                        self.colony.total_resources_collected = max(0, self.colony.total_resources_collected - self.resources) # Current colony loses its resources
                        
                        enemy.resources += self.resources # Enemy individual gains resources
                        enemy_colony_obj.total_resources_collected += self.resources # Enemy colony gains resources

                        if self in self.colony.individuals: # Remove self from colony
                            self.colony.individuals.remove(self)
                        current_cell.individual = None # Vacate current cell (individual is gone)
            # else: cell occupied by friendly, do nothing for now
        else: # Cell is not occupied by an individual
            resource_gain = 0
            claimed_new_territory = False
            if target_cell.colony_id is None: # Unclaimed cell
                resource_gain = 10 # Gain resources for claiming new cell
                target_cell.colony_id = self.colony.id
                claimed_new_territory = True
            elif target_cell.colony_id != self.colony.id: # Enemy territory (but no individual)
                if target_cell.colony_id < len(colonies_list) and colonies_list[target_cell.colony_id]: # Valid enemy colony
                    enemy_colony_obj = colonies_list[target_cell.colony_id]
                    if self.colony.fitness(grid) > enemy_colony_obj.fitness(grid): # If stronger, conquer
                        resource_gain = 10
                        # enemy_colony_obj.total_resources_collected = max(0, enemy_colony_obj.total_resources_collected - 5) # Optional penalty for losing territory
                        target_cell.colony_id = self.colony.id
                        claimed_new_territory = True
                    else: # Failed to conquer, don't move
                        return
                else: # Invalid enemy colony, treat as unclaimed
                    resource_gain = 10
                    target_cell.colony_id = self.colony.id
                    claimed_new_territory = True
            # else: Own territory, no resource gain from moving into empty own cell unless specific logic added

            if claimed_new_territory or target_cell.colony_id == self.colony.id : # If moved into own or newly claimed territory
                self.resources += resource_gain # Individual gains resources (e.g. from foraging)
                self.colony.total_resources_collected += resource_gain # Colony gets the resources
                
                current_cell.individual = None # Vacate current cell
                target_cell.individual = self # Occupy target cell
                self.x, self.y = x, y


# --- COLONY ---
class Colony:
    """
    Represents a colony in the simulation.

    Parameters
    ----------
    id : int
        The unique identifier for the colony.
    center_x : int
        The x-coordinate of the colony's initial center.
    center_y : int
        The y-coordinate of the colony's initial center.
    initial_color : tuple of int
        The RGB color for the colony's individuals.
    initial_light_color : tuple of int
        The RGB color for the colony's territory.

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
    total_resources_collected : int
        The total resources accumulated by the colony.
    """
    def __init__(self, id, center_x, center_y, initial_color, initial_light_color):
        self.id = id
        self.color = initial_color
        self.light_color = initial_light_color
        self.individuals = []
        # self.center = (center_x, center_y) # Initial center, less relevant if using center of mass
        self.total_resources_collected = 0 # Total resources accumulated by the colony

    def spawn_individual(self, x, y, grid):
        """
        Spawn a new individual at the specified location.

        Parameters
        ----------
        x : int
            The x-coordinate for the new individual.
        y : int
            The y-coordinate for the new individual.
        grid : list of list of Cell
            The simulation grid.
        """
        ind = Individual(x, y, self)
        self.individuals.append(ind)
        grid[y][x].individual = ind # Place individual on grid
        grid[y][x].colony_id = self.id # Mark cell as belonging to this colony

    def fitness(self, grid):
        """
        Compute the fitness of the colony based on area and resources.

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.

        Returns
        -------
        fitness : float
            The computed fitness value.
        """
        # Calculate area: number of cells owned by this colony
        area = sum(1 for row in grid for cell in row if cell.colony_id == self.id)
        # Fitness formula: weighted sum of area and resources
        return AREA_WEIGHT * area + RESOURCE_WEIGHT * self.total_resources_collected

    def reproduce(self, grid):
        """
        Attempt to reproduce by spawning a new individual if resources allow.

        Parameters
        ----------
        grid : list of list of Cell
            The simulation grid.
        """
        # Condition for reproduction: enough resources and at least one individual exists
        if self.total_resources_collected >= REPRODUCTION_THRESHOLD and self.individuals:
            spawn_candidates = [] # List of potential spawn locations
            for ind in self.individuals: # Check around each individual
                # Check current cell and adjacent cells (including diagonals)
                dirs = [(0,0), (0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
                random.shuffle(dirs)
                for dx, dy in dirs:
                    nx, ny = ind.x + dx, ind.y + dy
                    # Check if within grid and cell is empty
                    if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS and grid[ny][nx].individual is None:
                        # Prefer spawning in own territory or unclaimed territory
                        if grid[ny][nx].colony_id == self.id or grid[ny][nx].colony_id is None:
                             spawn_candidates.append((nx, ny))
            
            if spawn_candidates: # If suitable spawn locations found
                x, y = random.choice(spawn_candidates) # Pick one randomly
                self.spawn_individual(x, y, grid)
                self.total_resources_collected -= REPRODUCTION_COST # Pay reproduction cost
                self.total_resources_collected = max(0, self.total_resources_collected) # Ensure resources don't go negative


# --- DATA COLLECTOR ---
class DataCollector:
    """
    Collects and manages simulation data for plotting and saving.

    Attributes
    ----------
    time_steps : list of int
        The time steps at which data was collected.
    area_data : dict of int to list of int
        Area (number of cells) owned by each colony over time.
    population_data : dict of int to list of int
        Population (number of individuals) of each colony over time.
    resources_data : dict of int to list of int
        Resources collected by each colony over time.
    active_colony_ids : set of int
        Set of all colony IDs that ever existed.
    """
    def __init__(self):
        self.time_steps = []
        self.area_data = defaultdict(list)          # {colony_id: [area1, area2,...]}
        self.population_data = defaultdict(list)    # {colony_id: [pop1, pop2,...]}
        self.resources_data = defaultdict(list)     # {colony_id: [res1, res2,...]}
        self.active_colony_ids = set() # Keep track of all colony IDs that ever existed for plotting

    def collect_data(self, colonies_list, grid, current_step):
        """
        Collect data for all colonies at the current time step.

        Parameters
        ----------
        colonies_list : list of Colony
            List of all colonies in the simulation.
        grid : list of list of Cell
            The simulation grid.
        current_step : int
            The current time step.
        """
        self.time_steps.append(current_step)
        
        # Record data for all currently active colonies
        for colony in colonies_list:
            if not colony: continue # Skip if colony object is None (e.g. after potential removal)
            
            self.active_colony_ids.add(colony.id) # Add to set of all active IDs

            # 1. Area: Count cells belonging to this colony
            area = sum(1 for row in grid for cell in row if cell.colony_id == colony.id)
            self.area_data[colony.id].append(area)

            # 2. Population: Number of individuals in the colony
            population = len(colony.individuals)
            self.population_data[colony.id].append(population)

            # 3. Resources: Total resources of the colony
            resources = colony.total_resources_collected
            self.resources_data[colony.id].append(resources)
        
        # For any colony ID that was previously active but isn't in the current colonies_list
        # (e.g., died out), pad its data with the last known value or zero to maintain list lengths for plotting.
        # This example pads with the last value, assuming it persists until explicitly changed or the colony is gone.
        # More sophisticated handling (e.g. padding with 0 for dead colonies) can be added.
        all_ids_ever = self.active_colony_ids.copy()
        current_live_ids = {c.id for c in colonies_list if c}
        
        for cid in all_ids_ever:
            if cid not in current_live_ids: # Colony might have died
                if self.area_data[cid]: self.area_data[cid].append(self.area_data[cid][-1]) # Or append 0
                else: self.area_data[cid].append(0) # If it just died and had no data yet
                
                if self.population_data[cid]: self.population_data[cid].append(self.population_data[cid][-1]) # Or append 0
                else: self.population_data[cid].append(0)

                if self.resources_data[cid]: self.resources_data[cid].append(self.resources_data[cid][-1]) # Or append 0
                else: self.resources_data[cid].append(0)


    def plot_data(self, colors_list_global, output_dir="."):
        """
        Plot the collected data for area, population, and resources for each colony.

        Parameters
        ----------
        colors_list_global : list of tuple
            List of RGB color tuples for each colony.
        output_dir : str, optional
            Directory to save the plots (default is current directory).
        """
        if not self.time_steps:
            print("No data collected to plot.")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine max colony ID for color list extension
        max_colony_id = 0
        if self.active_colony_ids:
            max_colony_id = max(self.active_colony_ids)

        # Dynamically extend global color list if new colonies were added during simulation
        # This ensures plot colors can match simulation colors
        plot_colors = list(colors_list_global) # Make a mutable copy
        while len(plot_colors) <= max_colony_id:
            # Add new random colors (normalized for Matplotlib)
            plot_colors.append(tuple(np.random.randint(0, 256, size=3)/255.0))

        # --- Plot Area ---
        fig_area, ax_area = plt.subplots(figsize=(10, 6))
        ax_area.set_ylabel("Area (Cells)")
        for colony_id in sorted(list(self.active_colony_ids)): # Plot for all colonies that ever existed
            if colony_id in self.area_data and self.area_data[colony_id]:
                color_tuple = plot_colors[colony_id]
                # Ensure color is normalized (0-1 range)
                if any(c > 1 for c in color_tuple): color_tuple = tuple(c/255.0 for c in color_tuple)
                # Plot data up to the current number of time steps recorded for this colony
                ax_area.plot(self.time_steps[:len(self.area_data[colony_id])], self.area_data[colony_id],
                            label=f"Colony {colony_id}", color=color_tuple)
        ax_area.legend(loc="upper left")
        ax_area.grid(True)
        fig_area.suptitle("Area (Number of Cells) per Colony Over Time", fontsize=14)
        ax_area.set_xlabel("Time Step") # X-label for this specific plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        area_filename = os.path.join(output_dir, "colony_area_plot.png")
        fig_area.savefig(area_filename)
        print(f"Area plot saved to {area_filename}")

        # --- Plot Population ---
        fig_pop, ax_pop = plt.subplots(figsize=(10, 6))
        ax_pop.set_ylabel("Population (Individuals)")
        for colony_id in sorted(list(self.active_colony_ids)):
            if colony_id in self.population_data and self.population_data[colony_id]:
                color_tuple = plot_colors[colony_id]
                if any(c > 1 for c in color_tuple): color_tuple = tuple(c/255.0 for c in color_tuple)
                ax_pop.plot(self.time_steps[:len(self.population_data[colony_id])], self.population_data[colony_id],
                            label=f"Colony {colony_id}", color=color_tuple)
        ax_pop.legend(loc="upper left")
        ax_pop.grid(True)
        fig_pop.suptitle("Population per Colony Over Time", fontsize=14)
        ax_pop.set_xlabel("Time Step")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pop_filename = os.path.join(output_dir, "colony_population_plot.png")
        fig_pop.savefig(pop_filename)
        print(f"Population plot saved to {pop_filename}")

        # --- Plot Resources ---
        fig_res, ax_res = plt.subplots(figsize=(10, 6))
        ax_res.set_ylabel("Resources Collected")
        for colony_id in sorted(list(self.active_colony_ids)):
            if colony_id in self.resources_data and self.resources_data[colony_id]:
                color_tuple = plot_colors[colony_id]
                if any(c > 1 for c in color_tuple): color_tuple = tuple(c/255.0 for c in color_tuple)
                ax_res.plot(self.time_steps[:len(self.resources_data[colony_id])], self.resources_data[colony_id],
                            label=f"Colony {colony_id}", color=color_tuple)
        ax_res.legend(loc="upper left")
        ax_res.grid(True)
        fig_res.suptitle("Resources Collected per Colony Over Time", fontsize=14)
        ax_res.set_xlabel("Time Step")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        res_filename = os.path.join(output_dir, "colony_resources_plot.png")
        fig_res.savefig(res_filename)
        print(f"Resources plot saved to {res_filename}")

        plt.show() # Display all figures

    # --- New method to save data to CSV ---
    def save_data_to_csv(self, filename="colony_heuristics_data.csv"):
        """
        Save the collected data to a CSV file.

        Parameters
        ----------
        filename : str, optional
            The name of the CSV file to save (default is 'colony_heuristics_data.csv').
        """
        if not self.time_steps:
            print("No data collected to save.")
            return

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Create header row
            header = ["Time Step"]
            # Sort colony IDs to ensure consistent column order
            sorted_colony_ids = sorted(list(self.active_colony_ids))
            for colony_id in sorted_colony_ids:
                header.append(f"Colony {colony_id} Area")
                header.append(f"Colony {colony_id} Population")
                header.append(f"Colony {colony_id} Resources")
            writer.writerow(header)

            # Write data rows
            for i, time_step in enumerate(self.time_steps):
                row = [time_step]
                for colony_id in sorted_colony_ids:
                    # Get data for the current time step for each heuristic,
                    # padding with 0 if a colony didn't exist at this step (handled in collect_data)
                    area = self.area_data.get(colony_id, [0]*len(self.time_steps))[i]
                    population = self.population_data.get(colony_id, [0]*len(self.time_steps))[i]
                    resources = self.resources_data.get(colony_id, [0]*len(self.time_steps))[i]
                    row.extend([area, population, resources])
                writer.writerow(row)

        print(f"Data saved to {filename}")


# --- INIT GRID & COLONIES ---
grid = [[Cell(x, y) for x in range(GRID_DIMS)] for y in range(GRID_DIMS)]
colonies = [] # List to store Colony objects

# Create initial colonies as defined by NUM_COLONIES
for i in range(NUM_COLONIES):
    # Find a random, empty starting position for the colony center
    while True:
        cx, cy = random.randint(0, GRID_DIMS-1), random.randint(0, GRID_DIMS-1)
        if grid[cy][cx].colony_id is None and grid[cy][cx].individual is None:
            break # Found a free spot
    
    # Create the colony object with its unique ID and assigned colors
    colony = Colony(i, cx, cy, BASE_COLORS[i], LIGHT_COLORS[i])
    colonies.append(colony)
    
    # Spawn initial individuals for this colony
    if grid[cy][cx].individual is None: # Spawn first individual at colony center
         colony.spawn_individual(cx, cy, grid)
    
    # Spawn remaining initial individuals near the center
    for _ in range(NUM_INITIAL_INDIVIDUALS - 1):
        spawned = False
        attempts = 0 # Limit attempts to find a nearby spot
        while not spawned and attempts < 20:
            dirs = [(0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
            dx, dy = random.choice(dirs)
            nx, ny = cx + dx, cy + dy # Candidate spawn location
            # Check if within grid and cell is empty
            if 0 <= nx < GRID_DIMS and 0 <= ny < GRID_DIMS and grid[ny][nx].individual is None:
                colony.spawn_individual(nx, ny, grid)
                spawned = True
            attempts +=1
        if not spawned: # Fallback: if couldn't find nearby, try any random empty cell
            empty_cells_for_colony = []
            for r_idx, row_val in enumerate(grid):
                for c_idx, cell_in_grid in enumerate(row_val):
                    # Cell is empty and either unclaimed or already belongs to this colony
                    if cell_in_grid.individual is None and \
                       (cell_in_grid.colony_id is None or cell_in_grid.colony_id == colony.id):
                        empty_cells_for_colony.append((c_idx, r_idx))
            if empty_cells_for_colony:
                s_x, s_y = random.choice(empty_cells_for_colony)
                colony.spawn_individual(s_x, s_y, grid)


# --- DATA COLLECTION SETUP ---
data_collector = DataCollector()
current_time_step = 0

# --- MAIN LOOP ---
step_counter = 0
paused = False
running = True
while running:
    screen.fill((30,30,30)) # Dark grey background

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Allow mouse interaction only if simulation is not running for a fixed duration
        if SIMULATION_DURATION_STEPS is None:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos # Mouse click position
                grid_x, grid_y = mx // CELL_SIZE, my // CELL_SIZE # Convert to grid coordinates

                if 0 <= grid_x < GRID_DIMS and 0 <= grid_y < GRID_DIMS: # Click within grid
                    if event.button == 1:  # Left click - spawn new colony
                        # Can only spawn if cell is completely empty (no individual, no territory)
                        if grid[grid_y][grid_x].individual is None and grid[grid_y][grid_x].colony_id is None:
                            new_colony_id = len(colonies) # New ID is current number of colonies
                            
                            # Dynamically add colors for the new colony
                            new_base_color = tuple(np.random.randint(0, 256, size=3))
                            new_light_color = tuple(int(c*0.8) for c in new_base_color)
                            BASE_COLORS.append(new_base_color) # Add to global list for Pygame
                            LIGHT_COLORS.append(new_light_color)

                            new_colony = Colony(new_colony_id, grid_x, grid_y, new_base_color, new_light_color)
                            colonies.append(new_colony)
                            new_colony.spawn_individual(grid_x, grid_y, grid) # Spawn its first individual
                            print(f"Spawned new Colony {new_colony_id} at ({grid_x},{grid_y})")

                    elif event.button == 3: # Right click - clear area (destroy entities)
                        radius = 3 # Radius of destruction around click
                        for r_dx in range(-radius, radius + 1):
                            for r_dy in range(-radius, radius + 1):
                                if r_dx**2 + r_dy**2 <= radius**2: # Circular area
                                    check_x, check_y = grid_x + r_dx, grid_y + r_dy
                                    if 0 <= check_x < GRID_DIMS and 0 <= check_y < GRID_DIMS:
                                        cell_to_clear = grid[check_y][check_x]
                                        if cell_to_clear.individual: # If an individual is present
                                            ind = cell_to_clear.individual
                                            if ind.colony and ind in ind.colony.individuals:
                                                ind.colony.individuals.remove(ind) # Remove from its colony
                                            cell_to_clear.individual = None # Remove from cell
                                        # Optionally, penalize colony if its territory is destroyed
                                        # if cell_to_clear.colony_id is not None and colonies[cell_to_clear.colony_id]:
                                        #    colonies[cell_to_clear.colony_id].total_resources_collected = max(0, colonies[cell_to_clear.colony_id].total_resources_collected - 2)
                                        cell_to_clear.colony_id = None # Clear territory claim


    # --- Game Logic Update ---
    for colony in colonies: # Iterate through all colonies
        if colony: # Ensure colony object is valid
            # Iterate on a copy of individuals list, as it can be modified during moves/fights
            for ind in colony.individuals[:]:
                if ind in colony.individuals: # Check if individual still exists (not removed by a previous fight)
                    ind.move(grid, colonies)
            colony.reproduce(grid) # Attempt reproduction for the colony
    
    # Optional: Cleanup for "dead" colonies (no individuals and no territory)
    # This part is commented out as removing colonies from the list can complicate ID management
    # for data plotting if not handled carefully (e.g., by using a dictionary for colonies or marking as inactive).
    # For now, "dead" colonies will just show 0s in plots.


    # --- Data Collection ---
    data_collector.collect_data(colonies, grid, current_time_step)
    current_time_step += 1

    # --- Drawing ---
    # Draw grid cells and colony territories
    for y_idx in range(GRID_DIMS):
        for x_idx in range(GRID_DIMS):
            cell = grid[y_idx][x_idx]
            rect = pygame.Rect(x_idx*CELL_SIZE, y_idx*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # Draw territory color (light)
            if cell.colony_id is not None and cell.colony_id < len(LIGHT_COLORS):
                pygame.draw.rect(screen, LIGHT_COLORS[cell.colony_id], rect)
            # Draw individual color (base) on top if present
            if cell.individual and cell.individual.colony.id < len(BASE_COLORS):
                pygame.draw.rect(screen, BASE_COLORS[cell.individual.colony.id], rect, border_radius=3) # Slightly rounded individuals
    
    # Optional: Draw lines from colony center of mass to its individuals
    for colony in colonies:
        if colony and colony.individuals:
            # Calculate center of mass for the colony
            com_x = sum(ind.x for ind in colony.individuals) / len(colony.individuals)
            com_y = sum(ind.y for ind in colony.individuals) / len(colony.individuals)
            center_pixel_x = int(com_x * CELL_SIZE + CELL_SIZE / 2)
            center_pixel_y = int(com_y * CELL_SIZE + CELL_SIZE / 2)

            for ind in colony.individuals: # Draw line to each individual
                ind_pixel_x = int(ind.x * CELL_SIZE + CELL_SIZE / 2)
                ind_pixel_y = int(ind.y * CELL_SIZE + CELL_SIZE / 2)
                if colony.id < len(BASE_COLORS): # Ensure color is available
                    pygame.draw.line(screen, BASE_COLORS[colony.id],
                                     (center_pixel_x, center_pixel_y),
                                     (ind_pixel_x, ind_pixel_y), 1)

    # Display colony info (resources, population) on screen
    font = pygame.font.Font(None, 24) # Default Pygame font
    for i, colony in enumerate(colonies):
        if colony:
            text = f"Colony {colony.id}: Res={colony.total_resources_collected}, Pop={len(colony.individuals)}"
            text_surface = font.render(text, True, (255, 255, 255)) # White text
            screen.blit(text_surface, (10, 10 + i * 20)) # Position text for each colony


    pygame.display.flip() # Update the full display
    clock.tick(FPS) # Control simulation speed

    # Check for fixed duration simulation end
    if SIMULATION_DURATION_STEPS is not None and current_time_step >= SIMULATION_DURATION_STEPS:
        running = False


pygame.quit() # Uninitialize Pygame modules

# --- DATA SAVING ---
# Save data to CSV after the simulation ends
data_collector.save_data_to_csv("colony_simulation_data.csv")

# The plotting code is removed as per your request.
# data_collector.plot_data(BASE_COLORS, output_dir=".")