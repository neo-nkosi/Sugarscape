import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from collections import deque
import argparse
import math

class Agent:
    def __init__(self, id, x, y, sugar, metabolism, vision, broadcast_radius):
        self.id = id
        self.x = x
        self.y = y
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision
        self.broadcast_radius = broadcast_radius
        self.messages = deque(maxlen=100)
        self.destination = None

    def move(self, environment):
        x, y = self.x, self.y

        # Check messages for better locations
        best_message = max(self.messages, key=lambda m: m['sugar_amount'], default=None)
        if best_message and best_message['sugar_amount'] > environment.get_visible_sugar(self).sum():
            self.destination = (best_message['x'], best_message['y'])
        else:
            self.destination = None

        possible_moves = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < environment.width and 0 <= new_y < environment.height:
                    if (new_x, new_y) not in environment.agent_positions:
                        sugar_amount = environment.sugar[new_y, new_x]
                        possible_moves.append((new_x, new_y, sugar_amount))

        if not possible_moves:
            return  # No valid moves available

        # Add exploration chance
        if random.random() < 0.1:  # 10% chance to make a random move
            new_x, new_y, _ = random.choice(possible_moves)
        else:
            if self.destination:
                # Move towards the destination
                dest_x, dest_y = self.destination
                possible_moves.sort(key=lambda m: abs(m[0] - dest_x) + abs(m[1] - dest_y))
            else:
                # Move to the best sugar patch
                possible_moves.sort(key=lambda m: m[2], reverse=True)

            # Choose randomly from the top 3 best moves (or all if less than 3)
            best_moves = possible_moves[:min(3, len(possible_moves))]
            new_x, new_y, _ = random.choice(best_moves)

        environment.agent_positions.remove((self.x, self.y))
        self.x, self.y = new_x, new_y
        environment.agent_positions.add((new_x, new_y))

        if self.destination and (new_x, new_y) == self.destination:
            self.destination = None

    def broadcast_message(self, environment, timestep):
        visible_sugar = environment.get_visible_sugar(self).sum()
        message = {
            'sender_id': self.id,
            'sugar_amount': visible_sugar,
            'timestep': timestep,
            'x': self.x,
            'y': self.y
        }

        # Use spatial grid for efficient neighbor search
        grid_coords = environment.get_grid_cells_in_radius(self.x, self.y, self.broadcast_radius)
        nearby_agents = set()
        for coord in grid_coords:
            nearby_agents.update(environment.spatial_grid.get(coord, []))

        for other_agent in nearby_agents:
            if other_agent.id != self.id:
                distance = np.hypot(self.x - other_agent.x, self.y - other_agent.y)
                if distance <= self.broadcast_radius:
                    other_agent.messages.append(message)

class SugarscapeEnvironment:
    def __init__(self, width, height, cell_size=10, params=None, seed=None):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Set the random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

        # Environment parameters
        self.params = params if params else {}

        self.sugar = np.zeros((self.height, self.width), dtype=int)
        self.job_centers = []
        self.create_initial_sugar_peaks()
        self.max_sugar_landscape = self.sugar.copy()
        self.agents = self.initialize_agents()
        self.agent_positions = set((agent.x, agent.y) for agent in self.agents)
        self.dead_agents = []

        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption(self.params.get('title', 'Sugarscape Simulation'))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 10)
        self.running = True

        # Data tracking
        self.population_history = []
        self.average_wealth_history = []
        self.gini_coefficient_history = []
        self.employment_rate_history = []
        self.sugar_to_agent_ratio_history = []
        self.average_network_size_history = []
        self.active_sugar_peaks_history = []
        self.timestep = 0

        # Visualization toggles
        self.show_broadcast_radius = False
        self.show_agent_paths = False

        # Spatial grid parameters
        self.grid_cell_size = 5  # Adjust as needed
        self.spatial_grid = {}

    def create_initial_sugar_peaks(self, num_peaks=2):
        for _ in range(num_peaks):
            self.create_sugar_peak()
        self.update_sugar_landscape()

    def create_sugar_peak(self):
        x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
        duration = np.random.randint(*self.params['job_center_duration'])
        self.job_centers.append({
            'x': x, 'y': y,
            'duration': duration,
            'max_sugar': self.params['max_sugar']
        })

    def update_sugar_landscape(self):
        self.sugar = np.zeros((self.height, self.width))
        for center in self.job_centers:
            x_grid, y_grid = np.meshgrid(np.arange(self.width), np.arange(self.height))
            distance = np.sqrt((x_grid - center['x']) ** 2 + (y_grid - center['y']) ** 2)
            sugar_level = center['max_sugar'] * np.exp(-distance ** 2 / (2 * self.params['sugar_peak_spread'] ** 2))
            self.sugar += sugar_level
        self.sugar = np.clip(self.sugar, 0, self.params['max_sugar'])
        # Round sugar levels to nearest integer
        self.sugar = np.round(self.sugar).astype(int)

    def initialize_agents(self):
        agents = []
        available_positions = set((x, y) for x in range(self.width) for y in range(self.height))
        for i in range(self.params['num_agents']):
            if not available_positions:
                break
            x, y = available_positions.pop()
            agents.append(self.create_agent(i, x, y))
        return agents

    def create_agent(self, id, x, y):
        return Agent(
            id=id,
            x=x,
            y=y,
            sugar=np.random.randint(80, 120),
            metabolism=np.random.randint(*self.params['metabolism_range']),
            vision=np.random.randint(1, self.params['vision_range'] + 1),
            broadcast_radius=max(1, int(np.random.normal(
                self.params['avg_broadcast_radius'],
                self.params['avg_broadcast_radius'] / 3)))
        )

    def get_visible_sugar(self, agent):
        x, y = agent.x, agent.y
        vision = agent.vision
        x_min = max(0, x - vision)
        x_max = min(self.width, x + vision + 1)
        y_min = max(0, y - vision)
        y_max = min(self.height, y + vision + 1)
        visible_area = self.sugar[y_min:y_max, x_min:x_max]
        return visible_area

    def update_spatial_grid(self):
        self.spatial_grid.clear()
        for agent in self.agents:
            grid_x = agent.x // self.grid_cell_size
            grid_y = agent.y // self.grid_cell_size
            grid_coord = (grid_x, grid_y)
            if grid_coord not in self.spatial_grid:
                self.spatial_grid[grid_coord] = []
            self.spatial_grid[grid_coord].append(agent)

    def get_grid_cells_in_radius(self, x, y, radius):
        grid_radius = int(math.ceil(radius / self.grid_cell_size))
        grid_x = x // self.grid_cell_size
        grid_y = y // self.grid_cell_size
        cells = []
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                cells.append((grid_x + dx, grid_y + dy))
        return cells

    def collect_data(self, employed_agents):
        population = len(self.agents)
        total_wealth = sum(agent.sugar for agent in self.agents)
        average_wealth = total_wealth / population if population > 0 else 0

        # Calculate employment rate
        employment_rate = employed_agents / population if population > 0 else 0

        # Calculate sugar-to-agent ratio
        total_sugar = np.sum(self.sugar)
        sugar_to_agent_ratio = total_sugar / population if population > 0 else 0

        # Calculate average network size
        total_network_size = 0
        for agent in self.agents:
            network_size = len([other_agent for other_agent in self.agents
                                if other_agent != agent and
                                np.hypot(agent.x - other_agent.x, agent.y - other_agent.y) <= agent.broadcast_radius])
            total_network_size += network_size
        average_network_size = total_network_size / population if population > 0 else 0

        # Count active sugar peaks
        active_sugar_peaks = len(self.job_centers)

        self.population_history.append(population)
        self.average_wealth_history.append(average_wealth)
        self.gini_coefficient_history.append(self.calculate_gini_coefficient())
        self.employment_rate_history.append(employment_rate)
        self.sugar_to_agent_ratio_history.append(sugar_to_agent_ratio)
        self.average_network_size_history.append(average_network_size)
        self.active_sugar_peaks_history.append(active_sugar_peaks)

    def calculate_gini_coefficient(self):
        if not self.agents:
            return 0
        wealth_values = sorted(agent.sugar for agent in self.agents)
        cumulative_wealth = np.cumsum(wealth_values)
        n = len(wealth_values)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * wealth_values)) / (n * np.sum(wealth_values))
        return gini

    def render(self):
        self.screen.fill((255, 255, 255))

        for y in range(self.height):
            for x in range(self.width):
                sugar_level = self.sugar[y, x]
                color = self.get_color(sugar_level)
                pygame.draw.rect(self.screen, color,
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        for dead_agent in self.dead_agents:
            pygame.draw.circle(self.screen, (128, 128, 128),
                               (int(dead_agent['x'] * self.cell_size + self.cell_size / 2),
                                int(dead_agent['y'] * self.cell_size + self.cell_size / 2)),
                               int(self.cell_size / 3))

        for agent in self.agents:
            if self.show_broadcast_radius and agent.broadcast_radius:
                pygame.draw.circle(self.screen, (200, 200, 200),
                                   (int(agent.x * self.cell_size + self.cell_size / 2),
                                    int(agent.y * self.cell_size + self.cell_size / 2)),
                                   int(agent.broadcast_radius * self.cell_size), 1)

            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(agent.x * self.cell_size + self.cell_size / 2),
                                int(agent.y * self.cell_size + self.cell_size / 2)),
                               int(self.cell_size / 3))

            if self.show_agent_paths and agent.destination:
                pygame.draw.line(self.screen, (0, 255, 0),
                                 (int(agent.x * self.cell_size + self.cell_size / 2),
                                  int(agent.y * self.cell_size + self.cell_size / 2)),
                                 (int(agent.destination[0] * self.cell_size + self.cell_size / 2),
                                  int(agent.destination[1] * self.cell_size + self.cell_size / 2)),
                                 1)

        pygame.display.flip()

    def get_color(self, sugar_level):
        if sugar_level == 0:
            return (255, 255, 255)
        else:
            intensity = sugar_level / self.params['max_sugar']
            return (255, 255, int(255 * (1 - intensity)))

    def plot_lorenz_curve(self):
        if not self.agents:
            return
        wealth_values = sorted(agent.sugar for agent in self.agents)
        cumulative_wealth = np.cumsum(wealth_values)
        lorenz_curve = cumulative_wealth / cumulative_wealth[-1]
        plt.figure(figsize=(10, 6))
        plt.plot([0] + list(np.arange(1, len(wealth_values) + 1) / len(wealth_values)),
                 [0] + list(lorenz_curve))
        plt.plot([0, 1], [0, 1], 'r--')  # Line of perfect equality
        plt.title(f"Lorenz Curve (Timestep: {self.timestep}, {self.params.get('title', 'Sugarscape')})")
        plt.xlabel('Cumulative Share of Agents')
        plt.ylabel('Cumulative Share of Wealth')
        plt.grid(True)
        plt.show()

    def plot_all_metrics(self):
        fig, axs = plt.subplots(4, 2, figsize=(15, 25))
        fig.suptitle(f"{self.params.get('title', 'Sugarscape')} Simulation Metrics", fontsize=16)

        axs[0, 0].plot(self.population_history)
        axs[0, 0].set_title('Population over Time')
        axs[0, 0].set_xlabel('Timestep')
        axs[0, 0].set_ylabel('Population')
        axs[0, 0].set_ylim(bottom=0)

        axs[0, 1].plot(self.average_wealth_history)
        axs[0, 1].set_title('Average Wealth over Time')
        axs[0, 1].set_xlabel('Timestep')
        axs[0, 1].set_ylabel('Average Wealth')
        axs[0, 1].set_ylim(bottom=0)

        axs[1, 0].plot(self.gini_coefficient_history)
        axs[1, 0].set_title('Gini Coefficient over Time')
        axs[1, 0].set_xlabel('Timestep')
        axs[1, 0].set_ylabel('Gini Coefficient')
        axs[1, 0].set_ylim(0, 1)

        axs[1, 1].plot(self.employment_rate_history)
        axs[1, 1].set_title('Employment Rate over Time')
        axs[1, 1].set_xlabel('Timestep')
        axs[1, 1].set_ylabel('Employment Rate')
        axs[1, 1].set_ylim(0, 1)

        axs[2, 0].plot(self.sugar_to_agent_ratio_history)
        axs[2, 0].set_title('Sugar-to-Agent Ratio over Time')
        axs[2, 0].set_xlabel('Timestep')
        axs[2, 0].set_ylabel('Sugar-to-Agent Ratio')
        axs[2, 0].set_ylim(bottom=0)

        axs[2, 1].plot(self.average_network_size_history)
        axs[2, 1].set_title('Average Network Size over Time')
        axs[2, 1].set_xlabel('Timestep')
        axs[2, 1].set_ylabel('Average Network Size')
        axs[2, 1].set_ylim(bottom=0)

        axs[3, 0].plot(self.active_sugar_peaks_history)
        axs[3, 0].set_title('Active Sugar Peaks over Time')
        axs[3, 0].set_xlabel('Timestep')
        axs[3, 0].set_ylabel('Number of Active Sugar Peaks')
        axs[3, 0].set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    self.show_broadcast_radius = not self.show_broadcast_radius
                elif event.key == pygame.K_p:
                    self.show_agent_paths = not self.show_agent_paths
            elif event.type == pygame.QUIT:
                self.running = False

    def run_simulation(self, max_timesteps=1000):
        while self.running and self.timestep < max_timesteps:
            self.handle_events()
            self.step()
            self.render()
            self.clock.tick(10)
        self.plot_all_metrics()
        self.plot_lorenz_curve()
        pygame.quit()

    def step(self):
        # Update job centers
        for center in self.job_centers:
            center['duration'] -= 1
        self.job_centers = [center for center in self.job_centers if center['duration'] > 0]
        if np.random.random() < self.params['sugar_peak_frequency']:
            self.create_sugar_peak()
        self.update_sugar_landscape()

        employed_agents = 0  # Track employed agents

        # Update spatial grid before agents move
        self.update_spatial_grid()

        for agent in self.agents:
            agent.move(self)
            collected_sugar = self.sugar[agent.y, agent.x]
            if collected_sugar > 0:
                employed_agents += 1  # Count as employed if sugar was collected
            agent.sugar += collected_sugar
            self.sugar[agent.y, agent.x] = 0

            agent.sugar -= agent.metabolism

        # Update spatial grid after agents have moved
        self.update_spatial_grid()

        for agent in self.agents:
            agent.broadcast_message(self, self.timestep)

        for agent in self.agents:
            agent.messages = deque(
                [msg for msg in agent.messages if self.timestep - msg['timestep'] <= self.params['message_expiry']],
                maxlen=100)

        alive_agents = []
        for agent in self.agents:
            if agent.sugar <= 0:
                self.dead_agents.append({'x': agent.x, 'y': agent.y, 'death_time': self.timestep})
                self.agent_positions.remove((agent.x, agent.y))
            else:
                alive_agents.append(agent)
        self.agents = alive_agents

        self.dead_agents = [agent for agent in self.dead_agents if self.timestep - agent['death_time'] <= 5]

        self.collect_data(employed_agents)
        self.timestep += 1

def main():
    # Parameters for the rural environment
    rural_params = {
        'title': 'Rural Sugarscape Simulation',
        'num_agents': 500,
        'max_sugar': 4,
        'growth_rate': 1,
        'vision_range': 2,
        'avg_broadcast_radius': 4,
        'message_expiry': 15,
        'max_relay_messages': 5,
        'sugar_peak_frequency': 0.07,
        'sugar_peak_spread': 4,
        'job_center_duration': (30, 60),
        'metabolism_range': (1, 3)
    }

    # You can set the seed here for reproducibility
    seed = 42  # Replace with any integer or None for random behavior

    # Create and run the simulation
    env = SugarscapeEnvironment(width=50, height=50, params=rural_params, seed=seed)
    env.run_simulation(max_timesteps=1000)

if __name__ == "__main__":
    main()
