import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from collections import deque
import argparse

class Agent:
    def __init__(self, id, x, y, sugar, metabolism, vision, broadcast_radius=None):
        self.id = id
        self.x = x
        self.y = y
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision
        self.broadcast_radius = broadcast_radius
        self.messages = deque(maxlen=20)
        self.destination = None

    def move(self, environment):
        x, y = self.x, self.y
        vision = self.vision

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
                        distance_from_others = min(
                            (abs(new_x - a.x) + abs(new_y - a.y) for a in environment.agents if a != self),
                            default=0)
                        possible_moves.append((new_x, new_y, sugar_amount, distance_from_others))

        if not possible_moves:
            return  # No valid moves available

        # Add exploration chance
        if random.random() < 0.1:  # 10% chance to make a random move
            new_x, new_y, _, _ = random.choice(possible_moves)
        else:
            if self.destination:
                # Move towards the destination
                dest_x, dest_y = self.destination
                possible_moves.sort(
                    key=lambda m: abs(m[0] - dest_x) + abs(m[1] - dest_y) - 0.1 * m[3])
            else:
                # Move to the best sugar patch
                possible_moves.sort(key=lambda m: m[2] + 0.1 * m[3], reverse=True)

            # Choose randomly from the top 3 best moves (or all if less than 3)
            best_moves = possible_moves[:min(3, len(possible_moves))]
            new_x, new_y, _, _ = random.choice(best_moves)

        environment.agent_positions.remove((self.x, self.y))
        self.x, self.y = new_x, new_y
        environment.agent_positions.add((new_x, new_y))

        if self.destination and (new_x, new_y) == self.destination:
            self.destination = None

    def broadcast_message(self, environment, timestep):
        if self.broadcast_radius is None:
            return  # Agent doesn't have broadcasting capability
        visible_sugar = environment.get_visible_sugar(self).sum()
        message = {
            'sender_id': self.id,
            'sugar_amount': visible_sugar,
            'timestep': timestep,
            'x': self.x,
            'y': self.y
        }
        for other_agent in environment.agents:
            if other_agent.id != self.id:
                distance = np.sqrt((self.x - other_agent.x) ** 2 + (self.y - other_agent.y) ** 2)
                if distance <= self.broadcast_radius:
                    other_agent.messages.append(message)

class BaseSugarscapeEnvironment:
    def __init__(self, width, height, num_agents, cell_size=10, max_timesteps=1000,
                 show_sugar_levels=False, seed=None, **kwargs):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.cell_size = cell_size
        self.max_timesteps = max_timesteps
        self.show_sugar_levels = show_sugar_levels

        self.params = {
            'max_sugar': 5,
            'growth_rate': 1,
            'sugar_peak_frequency': 0.04,
            'sugar_peak_spread': 6,
            'job_center_duration': (40, 100),
            'vision_range': 1,
            'message_expiry': 15,
            'max_relay_messages': 10
        }

        self.job_centers = []
        self.sugar = np.zeros((self.height, self.width), dtype=int)
        self.create_initial_sugar_peaks()
        self.max_sugar_landscape = self.sugar.copy()
        self.agents = self.initialize_agents()
        self.agent_positions = set((agent.x, agent.y) for agent in self.agents)
        self.dead_agents = []

        pygame.init()
        window_width = self.width * self.cell_size
        window_height = self.height * self.cell_size
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Sugarscape Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 10)
        self.running = True

        self.population_history = []
        self.average_wealth_history = []
        self.gini_coefficient_history = []
        self.timestep = 0

        # Visualization toggles
        self.show_broadcast_radius = kwargs.get('show_broadcast_radius', False)
        self.show_agent_paths = kwargs.get('show_agent_paths', False)

    def create_initial_sugar_peaks(self, num_peaks=2):
        for _ in range(num_peaks):
            self.create_job_center()
        self.update_sugar_landscape()

    def create_job_center(self):
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
        self.sugar = np.round(self.sugar).astype(int)

    def get_visible_sugar(self, agent):
        x, y = agent.x, agent.y
        vision = agent.vision
        x_min = max(0, x - vision)
        x_max = min(self.width, x + vision + 1)
        y_min = max(0, y - vision)
        y_max = min(self.height, y + vision + 1)
        visible_area = self.sugar[y_min:y_max, x_min:x_max]
        return visible_area

    def collect_data(self):
        population = len(self.agents)
        total_wealth = sum(agent.sugar for agent in self.agents)
        average_wealth = total_wealth / population if population > 0 else 0

        self.population_history.append(population)
        self.average_wealth_history.append(average_wealth)
        self.gini_coefficient_history.append(self.calculate_gini_coefficient())

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

                if self.show_sugar_levels:
                    sugar_text = self.font.render(f"{sugar_level}", True, (0, 0, 0))
                    text_rect = sugar_text.get_rect(center=(x * self.cell_size + self.cell_size // 2,
                                                            y * self.cell_size + self.cell_size // 2))
                    self.screen.blit(sugar_text, text_rect)

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

    def plot_results(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.plot(self.population_history)
        plt.title('Population over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Population')

        plt.subplot(132)
        plt.plot(self.average_wealth_history)
        plt.title('Average Wealth over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Average Wealth')

        plt.subplot(133)
        plt.plot(self.gini_coefficient_history)
        plt.title('Gini Coefficient over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Gini Coefficient')

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

    def run_simulation(self):
        while self.running and self.timestep < self.max_timesteps:
            self.handle_events()
            self.step()
            self.render()
            self.clock.tick(10)
        self.plot_results()

class SugarscapeEnvironmentWithCommunication(BaseSugarscapeEnvironment):
    def __init__(self, *args, **kwargs):
        self.broadcast_radius = kwargs.pop('broadcast_radius', 15)
        super().__init__(*args, **kwargs)

    def initialize_agents(self):
        agents = []
        available_positions = set((x, y) for x in range(self.width) for y in range(self.height))
        for i in range(self.num_agents):
            if not available_positions:
                break
            x, y = available_positions.pop()
            agents.append(Agent(
                id=i,
                x=x,
                y=y,
                sugar=np.random.randint(40, 80),
                metabolism=np.random.randint(1, 3),
                vision=np.random.randint(1, self.params['vision_range'] + 1),
                broadcast_radius=max(1, int(np.random.normal(self.broadcast_radius, self.broadcast_radius / 3)))
            ))
        return agents

    def step(self):
        # Update job centers
        for center in self.job_centers:
            center['duration'] -= 1
        self.job_centers = [center for center in self.job_centers if center['duration'] > 0]
        if np.random.random() < self.params['sugar_peak_frequency']:
            self.create_job_center()
        self.update_sugar_landscape()

        for agent in self.agents:
            agent.move(self)
            collected_sugar = self.sugar[agent.y, agent.x]
            agent.sugar += collected_sugar
            self.sugar[agent.y, agent.x] = 0
            agent.sugar -= agent.metabolism
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

        self.collect_data()
        self.timestep += 1

class SugarscapeEnvironmentWithoutCommunication(BaseSugarscapeEnvironment):
    def initialize_agents(self):
        agents = []
        available_positions = set((x, y) for x in range(self.width) for y in range(self.height))
        for i in range(self.num_agents):
            if not available_positions:
                break
            x, y = available_positions.pop()
            agents.append(Agent(
                id=i,
                x=x,
                y=y,
                sugar=np.random.randint(40, 80),
                metabolism=np.random.randint(1, 3),
                vision=1  # Limited vision
            ))
        return agents

    def step(self):
        # Update job centers
        for center in self.job_centers:
            center['duration'] -= 1
        self.job_centers = [center for center in self.job_centers if center['duration'] > 0]
        if np.random.random() < self.params['sugar_peak_frequency']:
            self.create_job_center()
        self.update_sugar_landscape()

        for agent in self.agents:
            # Agent movement without communication
            x, y = agent.x, agent.y
            adjacent_cells = [(x + dx, y + dy) for dx in range(-agent.vision, agent.vision + 1)
                              for dy in range(-agent.vision, agent.vision + 1)
                              if (dx != 0 or dy != 0) and
                              0 <= x + dx < self.width and 0 <= y + dy < self.height and
                              (x + dx, y + dy) not in self.agent_positions]

            best_cell = None
            best_sugar = -1

            for cell_x, cell_y in adjacent_cells:
                sugar_amount = self.sugar[cell_y, cell_x]
                if sugar_amount > best_sugar:
                    best_sugar = sugar_amount
                    best_cell = (cell_x, cell_y)

            if best_cell and best_sugar > 0:
                self.agent_positions.remove((agent.x, agent.y))
                agent.x, agent.y = best_cell
                self.agent_positions.add((agent.x, agent.y))
            else:
                # Move randomly if no sugar is visible
                available_cells = [cell for cell in adjacent_cells if cell not in self.agent_positions]
                if available_cells:
                    new_x, new_y = random.choice(available_cells)
                    self.agent_positions.remove((agent.x, agent.y))
                    agent.x, agent.y = new_x, new_y
                    self.agent_positions.add((agent.x, agent.y))
                # If no available cells, the agent stays put

            collected_sugar = self.sugar[agent.y, agent.x]
            agent.sugar += collected_sugar
            self.sugar[agent.y, agent.x] = 0
            agent.sugar -= agent.metabolism

        alive_agents = []
        for agent in self.agents:
            if agent.sugar <= 0:
                self.dead_agents.append({'x': agent.x, 'y': agent.y, 'death_time': self.timestep})
                self.agent_positions.remove((agent.x, agent.y))
            else:
                alive_agents.append(agent)
        self.agents = alive_agents

        self.dead_agents = [agent for agent in self.dead_agents if self.timestep - agent['death_time'] <= 5]

        self.collect_data()
        self.timestep += 1

def parse_arguments():
    parser = argparse.ArgumentParser(description='Sugarscape Simulation Parameters')
    parser.add_argument('--width', type=int, default=50, help='Width of the environment')
    parser.add_argument('--height', type=int, default=50, help='Height of the environment')
    parser.add_argument('--num_agents', type=int, default=200, help='Number of agents')
    parser.add_argument('--cell_size', type=int, default=10, help='Size of each cell in pixels')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='Maximum number of timesteps')
    parser.add_argument('--broadcast_radius', type=int, default=15, help='Broadcast radius for agents')
    parser.add_argument('--show_sugar_levels', action='store_true', help='Display sugar levels on the grid')
    parser.add_argument('--show_broadcast_radius', action='store_true', help='Display broadcast radius of agents')
    parser.add_argument('--show_agent_paths', action='store_true', help='Display paths of agents towards destinations')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--communication', action='store_true', help='Enable agent communication')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.communication:
        environment = SugarscapeEnvironmentWithCommunication(
            width=args.width,
            height=args.height,
            num_agents=args.num_agents,
            cell_size=args.cell_size,
            max_timesteps=args.max_timesteps,
            show_sugar_levels=args.show_sugar_levels,
            show_broadcast_radius=args.show_broadcast_radius,
            show_agent_paths=args.show_agent_paths,
            broadcast_radius=args.broadcast_radius,
            seed=args.seed
        )
    else:
        environment = SugarscapeEnvironmentWithoutCommunication(
            width=args.width,
            height=args.height,
            num_agents=args.num_agents,
            cell_size=args.cell_size,
            max_timesteps=args.max_timesteps,
            show_sugar_levels=args.show_sugar_levels,
            seed=args.seed
        )
    environment.run_simulation()

if __name__ == "__main__":
    main()
