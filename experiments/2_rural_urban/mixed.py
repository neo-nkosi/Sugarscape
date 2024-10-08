import random
import numpy as np
import pygame
from collections import deque
import matplotlib.pyplot as plt

class SugarscapeEnvironment:
    def __init__(self, width, height, num_agents, setting='urban', cell_size=10, show_broadcast_radius=True, show_agent_paths=True):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.cell_size = cell_size
        self.setting = setting
        self.show_broadcast_radius = show_broadcast_radius
        self.show_agent_paths = show_agent_paths

        # Define parameter sets for urban and rural environments
        self.urban_params = {
            'max_sugar': 5,
            'growth_rate': 1,
            'vision_range': 2,
            'avg_broadcast_radius': 10,
            'message_expiry': 15,
            'max_relay_messages': 5,
            'sugar_peak_frequency': 0.05,  # x% chance of new sugar peak per step
            'sugar_peak_spread': 7,
            'job_center_duration': (80, 200)  # min and max duration for job centers
        }

        self.rural_params = {
            'max_sugar': 4,
            'growth_rate': 1,
            'vision_range': 2,
            'avg_broadcast_radius': 6,
            'message_expiry': 10,
            'max_relay_messages': 3,
            'sugar_peak_frequency': 0.07,  # 1% chance of new sugar peak per step
            'sugar_peak_spread': 4,
            'job_center_duration': (60, 160)  # min and max duration for job centers
        }

        # Set parameters based on the environment setting
        self.params = self.urban_params if setting == 'urban' else self.rural_params

        self.job_centers = []
        self.sugar = np.zeros((self.height, self.width))
        self.create_initial_sugar_peaks()  # Create initial sugar peaks
        self.max_sugar_landscape = self.sugar.copy()
        self.agents = self.initialize_agents()
        self.agent_positions = set((agent['x'], agent['y']) for agent in self.agents)
        self.dead_agents = []

        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption(f"Sugarscape Simulation - {setting.capitalize()} Setting")
        self.clock = pygame.time.Clock()

        # Data tracking
        self.population_history = []
        self.average_wealth_history = []
        self.gini_coefficient_history = []
        self.timestep = 0

    def create_initial_sugar_peaks(self, num_peaks=2):
        for _ in range(num_peaks):
            self.create_job_center()
        self.update_sugar_landscape()

    def create_job_center(self):
        x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
        duration = np.random.randint(*self.params['job_center_duration'])
        self.job_centers.append({
            'x': x,
            'y': y,
            'duration': duration,
            'max_sugar': self.params['max_sugar']
        })

    def update_sugar_landscape(self):
        self.sugar = np.zeros((self.height, self.width))
        for center in self.job_centers:
            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            distance = np.sqrt((x - center['x']) ** 2 + (y - center['y']) ** 2)
            sugar_level = center['max_sugar'] * np.exp(-distance ** 2 / (2 * self.params['sugar_peak_spread'] ** 2))
            self.sugar += sugar_level
        self.sugar = np.clip(self.sugar, 0, self.params['max_sugar'])

    def initialize_agents(self):
        agents = []
        available_positions = set((x, y) for x in range(self.width) for y in range(self.height))
        for i in range(self.num_agents):
            if not available_positions:
                break
            x, y = available_positions.pop()
            agents.append(self.create_agent(i, x, y))
        return agents

    def create_agent(self, id, x, y):
        return {
            'id': id,
            'x': x,
            'y': y,
            'sugar': np.random.randint(100, 150),
            'metabolism': np.random.randint(1, 4),
            'vision': np.random.randint(1, self.params['vision_range'] + 1),
            'broadcast_radius': max(1, int(np.random.normal(self.params['avg_broadcast_radius'], self.params['avg_broadcast_radius'] / 3))),
            'messages': deque(maxlen=100),
            'destination': None
        }

    def get_visible_sugar(self, agent):
        x, y = agent['x'], agent['y']
        vision = agent['vision']
        visible_area = self.sugar[max(0, y - vision):min(self.height, y + vision + 1),
                       max(0, x - vision):min(self.width, x + vision + 1)]
        return visible_area

    def get_top_sugar_locations(self, agent):
        return sorted(agent['messages'], key=lambda x: x['sugar_amount'], reverse=True)[:self.params['max_relay_messages']]

    def broadcast_message(self, agent, timestep):
        visible_sugar = self.get_visible_sugar(agent).sum()
        message = {
            'sender_id': agent['id'],
            'sugar_amount': visible_sugar,
            'timestep': timestep,
            'x': agent['x'],
            'y': agent['y']
        }
        top_locations = self.get_top_sugar_locations(agent)

        for other_agent in self.agents:
            if other_agent['id'] != agent['id']:
                distance = np.sqrt((agent['x'] - other_agent['x']) ** 2 + (agent['y'] - other_agent['y']) ** 2)
                if distance <= agent['broadcast_radius']:
                    other_agent['messages'].append(message)
                    for location in top_locations:
                        other_agent['messages'].append(location)

    def move_agent(self, agent):
        x, y = agent['x'], agent['y']
        vision = agent['vision']

        # Check messages for better locations
        best_message = max(agent['messages'], key=lambda m: m['sugar_amount'], default=None)
        if best_message and best_message['sugar_amount'] > self.get_visible_sugar(agent).sum():
            agent['destination'] = (best_message['x'], best_message['y'])
        else:
            agent['destination'] = None

        possible_moves = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # Added diagonal moves
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height and (new_x, new_y) not in self.agent_positions:
                sugar_amount = self.sugar[new_y, new_x]
                distance_from_others = min(
                    (abs(new_x - a['x']) + abs(new_y - a['y']) for a in self.agents if a != agent), default=0)
                possible_moves.append((new_x, new_y, sugar_amount, distance_from_others))

        if not possible_moves:
            return  # No valid moves available

        # Add exploration chance
        if random.random() < 0.1:  # 10% chance to make a random move
            new_x, new_y, _, _ = random.choice(possible_moves)
        else:
            if agent['destination']:
                # Move towards the destination
                dest_x, dest_y = agent['destination']
                possible_moves.sort(
                    key=lambda m: abs(m[0] - dest_x) + abs(m[1] - dest_y) - 0.1 * m[3])  # Slight preference for spacing
            else:
                # Move to the best sugar patch
                possible_moves.sort(key=lambda m: m[2] + 0.1 * m[3], reverse=True)  # Slight preference for spacing

            # Choose randomly from the top 3 best moves (or all if less than 3)
            best_moves = possible_moves[:min(3, len(possible_moves))]
            new_x, new_y, _, _ = random.choice(best_moves)

        self.agent_positions.remove((agent['x'], agent['y']))
        agent['x'], agent['y'] = new_x, new_y
        self.agent_positions.add((new_x, new_y))

        if agent['destination'] and (new_x, new_y) == agent['destination']:
            agent['destination'] = None

    def step(self, timestep):
        # Update job centers
        for center in self.job_centers:
            center['duration'] -= 1
        self.job_centers = [center for center in self.job_centers if center['duration'] > 0]
        if np.random.random() < self.params['sugar_peak_frequency']:
            self.create_job_center()
        self.update_sugar_landscape()

        for agent in self.agents:
            self.move_agent(agent)

            collected_sugar = self.sugar[agent['y'], agent['x']]
            agent['sugar'] += collected_sugar
            self.sugar[agent['y'], agent['x']] = 0

            agent['sugar'] -= agent['metabolism']
            self.broadcast_message(agent, timestep)

        for agent in self.agents:
            agent['messages'] = deque(
                [msg for msg in agent['messages'] if timestep - msg['timestep'] <= self.params['message_expiry']], maxlen=100)

        alive_agents = []
        for agent in self.agents:
            if agent['sugar'] <= 0:
                self.dead_agents.append({'x': agent['x'], 'y': agent['y'], 'death_time': timestep})
                self.agent_positions.remove((agent['x'], agent['y']))
            else:
                alive_agents.append(agent)
        self.agents = alive_agents

        self.dead_agents = [agent for agent in self.dead_agents if timestep - agent['death_time'] <= 5]

        # Collect data after each step
        self.collect_data()

        self.timestep += 1

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
            if self.show_broadcast_radius:
                pygame.draw.circle(self.screen, (200, 200, 200),
                                   (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                    int(agent['y'] * self.cell_size + self.cell_size / 2)),
                                   int(agent['broadcast_radius'] * self.cell_size), 1)

            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                int(agent['y'] * self.cell_size + self.cell_size / 2)),
                               int(self.cell_size / 3))

            if self.show_agent_paths and agent['destination']:
                pygame.draw.line(self.screen, (0, 255, 0),
                                 (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                  int(agent['y'] * self.cell_size + self.cell_size / 2)),
                                 (int(agent['destination'][0] * self.cell_size + self.cell_size / 2),
                                  int(agent['destination'][1] * self.cell_size + self.cell_size / 2)),
                                 1)

        # Draw grid lines
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height * self.cell_size))
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width * self.cell_size, y))

        pygame.display.flip()

    def get_color(self, sugar_level):
        if sugar_level == 0:
            return (255, 255, 255)
        else:
            intensity = sugar_level / self.params['max_sugar']
            return (255, 255, int(255 * (1 - intensity)))

    def collect_data(self):
        population = len(self.agents)
        total_wealth = sum(agent['sugar'] for agent in self.agents)
        average_wealth = total_wealth / population if population > 0 else 0

        self.population_history.append(population)
        self.average_wealth_history.append(average_wealth)
        self.gini_coefficient_history.append(self.calculate_gini_coefficient())

    def calculate_gini_coefficient(self):
        if not self.agents:
            return 0
        wealth_values = sorted(agent['sugar'] for agent in self.agents)
        cumulative_wealth = np.cumsum(wealth_values)
        return (np.sum((2 * np.arange(1, len(wealth_values) + 1) - len(wealth_values) - 1) * wealth_values) /
                (len(wealth_values) * np.sum(wealth_values)))

    def plot_population_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.population_history)
        plt.title(f'Agent Population over Time ({self.setting.capitalize()} Setting)')
        plt.xlabel('Timestep')
        plt.ylabel('Population')
        plt.grid(True)
        plt.show()

    def plot_average_wealth_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.average_wealth_history)
        plt.title(f'Average Agent Wealth over Time ({self.setting.capitalize()} Setting)')
        plt.xlabel('Timestep')
        plt.ylabel('Average Wealth')
        plt.grid(True)
        plt.show()

    def plot_lorenz_curve(self):
        if not self.agents:
            return
        wealth_values = sorted(agent['sugar'] for agent in self.agents)
        cumulative_wealth = np.cumsum(wealth_values)
        lorenz_curve = cumulative_wealth / cumulative_wealth[-1]
        plt.figure(figsize=(10, 6))
        plt.plot([0] + list(np.arange(1, len(wealth_values) + 1) / len(wealth_values)), [0] + list(lorenz_curve))
        plt.plot([0, 1], [0, 1], 'r--')  # Line of perfect equality
        plt.title(f'Lorenz Curve (Timestep: {self.timestep}, {self.setting.capitalize()} Setting)')
        plt.xlabel('Cumulative Share of Agents')
        plt.ylabel('Cumulative Share of Wealth')
        plt.grid(True)
        plt.show()

    def plot_gini_coefficient_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.gini_coefficient_history)
        plt.title(f'Gini Coefficient over Time ({self.setting.capitalize()} Setting)')
        plt.xlabel('Timestep')
        plt.ylabel('Gini Coefficient')
        plt.grid(True)
        plt.show()

    def run_simulation(self, max_timesteps=1000):
        running = True
        while running and self.timestep < max_timesteps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.step(self.timestep)
            self.render()
            self.clock.tick(5)

        # Plot the collected data
        self.plot_population_history()
        self.plot_average_wealth_history()
        self.plot_gini_coefficient_history()
        self.plot_lorenz_curve()


class MixedSugarscapeEnvironment(SugarscapeEnvironment):
    def __init__(self, width, height, num_agents, cell_size=5, show_broadcast_radius=True, show_agent_paths=True,
                 num_initial_peaks=4, sugar_peak_scale=1.5):
        self.urban_areas = [(0, width // 2, height // 2, height),
                            (width // 2, width, 0, height // 2)]
        self.rural_areas = [(0, width // 2, 0, height // 2),
                            (width // 2, width, height // 2, height)]
        self.num_initial_peaks = num_initial_peaks
        self.sugar_peak_scale = sugar_peak_scale
        super().__init__(width, height, num_agents, 'mixed', cell_size, show_broadcast_radius, show_agent_paths)
        pygame.display.set_caption("Sugarscape Simulation - Mixed Setting")

    def create_initial_sugar_peaks(self):
        for _ in range(self.num_initial_peaks):
            self.create_job_center()
        self.update_sugar_landscape()

    def is_urban(self, x, y):
        return any(x1 <= x < x2 and y1 <= y < y2 for x1, x2, y1, y2 in self.urban_areas)

    def create_agent(self, id, x, y):
        agent = super().create_agent(id, x, y)
        self.update_agent_parameters(agent)
        return agent

    def update_agent_parameters(self, agent):
        if self.is_urban(agent['x'], agent['y']):
            agent['broadcast_radius'] = max(1, int(np.random.normal(self.urban_params['avg_broadcast_radius'],
                                                                    self.urban_params['avg_broadcast_radius'] / 3)))
            agent['vision'] = np.random.randint(1, self.urban_params['vision_range'] + 1)
        else:
            agent['broadcast_radius'] = max(1, int(np.random.normal(self.rural_params['avg_broadcast_radius'],
                                                                    self.rural_params['avg_broadcast_radius'] / 3)))
            agent['vision'] = np.random.randint(1, self.rural_params['vision_range'] + 1)

    def move_agent(self, agent):
        old_x, old_y = agent['x'], agent['y']
        super().move_agent(agent)
        if (self.is_urban(old_x, old_y) != self.is_urban(agent['x'], agent['y'])):
            self.update_agent_parameters(agent)

    def create_job_center(self):
        x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
        params = self.urban_params if self.is_urban(x, y) else self.rural_params
        duration = np.random.randint(*params['job_center_duration'])
        self.job_centers.append({
            'x': x,
            'y': y,
            'duration': duration,
            'max_sugar': params['max_sugar'] * self.sugar_peak_scale
        })

    def render(self):
        self.screen.fill((255, 255, 255))

        for y in range(self.height):
            for x in range(self.width):
                sugar_level = self.sugar[y, x]
                color = self.get_color(sugar_level)
                pygame.draw.rect(self.screen, color,
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # Highlight urban areas
        for x1, x2, y1, y2 in self.urban_areas:
            surface = pygame.Surface(((x2 - x1) * self.cell_size, (y2 - y1) * self.cell_size))
            surface.set_alpha(50)  # Adjust transparency (0-255)
            surface.fill((130, 130, 255))  # Light blue for urban areas
            self.screen.blit(surface, (x1 * self.cell_size, y1 * self.cell_size))

        # Highlight rural areas
        for x1, x2, y1, y2 in self.rural_areas:
            surface = pygame.Surface(((x2 - x1) * self.cell_size, (y2 - y1) * self.cell_size))
            surface.set_alpha(50)  # Adjust transparency (0-255)
            surface.fill((255, 130, 130))  # Light red for rural areas
            self.screen.blit(surface, (x1 * self.cell_size, y1 * self.cell_size))

        for dead_agent in self.dead_agents:
            pygame.draw.circle(self.screen, (128, 128, 128),
                               (int(dead_agent['x'] * self.cell_size + self.cell_size / 2),
                                int(dead_agent['y'] * self.cell_size + self.cell_size / 2)),
                               max(1, int(self.cell_size / 3)))

        for agent in self.agents:
            if self.show_broadcast_radius:
                pygame.draw.circle(self.screen, (200, 200, 200),
                                   (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                    int(agent['y'] * self.cell_size + self.cell_size / 2)),
                                   max(1, int(agent['broadcast_radius'] * self.cell_size)), 1)

            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                int(agent['y'] * self.cell_size + self.cell_size / 2)),
                               max(1, int(self.cell_size / 3)))

            if self.show_agent_paths and agent['destination']:
                pygame.draw.line(self.screen, (0, 255, 0),
                                 (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                  int(agent['y'] * self.cell_size + self.cell_size / 2)),
                                 (int(agent['destination'][0] * self.cell_size + self.cell_size / 2),
                                  int(agent['destination'][1] * self.cell_size + self.cell_size / 2)),
                                 1)

        # Draw grid lines (optional, remove if too cluttered)
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height * self.cell_size))
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width * self.cell_size, y))

        pygame.display.flip()

    def run_simulation(self, max_timesteps=1000):
        super().run_simulation(max_timesteps)
        self.plot_urban_rural_comparison()

    def plot_urban_rural_comparison(self):
        urban_agents = [agent for agent in self.agents if self.is_urban(agent['x'], agent['y'])]
        rural_agents = [agent for agent in self.agents if not self.is_urban(agent['x'], agent['y'])]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist([agent['sugar'] for agent in urban_agents], bins=20, alpha=0.5, label='Urban')
        plt.hist([agent['sugar'] for agent in rural_agents], bins=20, alpha=0.5, label='Rural')
        plt.title('Wealth Distribution: Urban vs Rural')
        plt.xlabel('Sugar Amount')
        plt.ylabel('Number of Agents')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter([agent['x'] for agent in urban_agents], [agent['y'] for agent in urban_agents], c='r', alpha=0.5,
                    label='Urban')
        plt.scatter([agent['x'] for agent in rural_agents], [agent['y'] for agent in rural_agents], c='b', alpha=0.5,
                    label='Rural')
        plt.title('Agent Distribution')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Example usage with visualization options

mixed_env = MixedSugarscapeEnvironment(width=80, height=80, num_agents=400, cell_size=6, show_broadcast_radius=False,
                                       show_agent_paths=False,
                                       num_initial_peaks=10, sugar_peak_scale=1.5)

# Run simulations

mixed_env.run_simulation(max_timesteps=1000)

# Quit Pygame after all simulations are complete
pygame.quit()