import random
import numpy as np
import pygame
import matplotlib.pyplot as plt
from collections import deque

class SugarscapeEnvironment:
    def __init__(self, width, height, num_agents, cell_size=10, show_sugar_levels=True,
                 show_broadcast_radius=True, show_agent_paths=True, broadcast_radius=5):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.cell_size = cell_size
        self.show_sugar_levels = show_sugar_levels
        self.show_broadcast_radius = show_broadcast_radius
        self.show_agent_paths = show_agent_paths
        self.broadcast_radius = broadcast_radius

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
        self.agent_positions = set((agent['x'], agent['y']) for agent in self.agents)
        self.dead_agents = []

        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Sugarscape Simulation - With Broadcasting")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 10)

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
            'x': x, 'y': y,
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
        self.sugar = np.round(self.sugar).astype(int)

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
            'id': id, 'x': x, 'y': y,
            'sugar': np.random.randint(40, 80),
            'metabolism': np.random.randint(1, 3),
            'vision': np.random.randint(1, self.params['vision_range'] + 1),
            'broadcast_radius': max(1, int(np.random.normal(self.broadcast_radius, self.broadcast_radius / 3))),
            'messages': deque(maxlen=100),
            'destination': None
        }

    def get_adjacent_cells(self, x, y):
        adjacent = [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1),
        ]
        return [(x, y) for x, y in adjacent if 0 <= x < self.width and 0 <= y < self.height]

    def get_visible_sugar(self, agent):
        x, y = agent['x'], agent['y']
        vision = agent['vision']
        visible_area = self.sugar[max(0, y - vision):min(self.height, y + vision + 1),
                       max(0, x - vision):min(self.width, x + vision + 1)]
        return visible_area

    def broadcast_message(self, agent, timestep):
        visible_sugar = self.get_visible_sugar(agent).sum()
        message = {
            'sender_id': agent['id'],
            'sugar_amount': visible_sugar,
            'timestep': timestep,
            'x': agent['x'],
            'y': agent['y']
        }
        for other_agent in self.agents:
            if other_agent['id'] != agent['id']:
                distance = np.sqrt((agent['x'] - other_agent['x']) ** 2 + (agent['y'] - other_agent['y']) ** 2)
                if distance <= agent['broadcast_radius']:
                    other_agent['messages'].append(message)

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
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
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
                    key=lambda m: abs(m[0] - dest_x) + abs(m[1] - dest_y) - 0.1 * m[3])
            else:
                # Move to the best sugar patch
                possible_moves.sort(key=lambda m: m[2] + 0.1 * m[3], reverse=True)

            # Choose randomly from the top 3 best moves (or all if less than 3)
            best_moves = possible_moves[:min(3, len(possible_moves))]
            new_x, new_y, _, _ = random.choice(best_moves)

        self.agent_positions.remove((agent['x'], agent['y']))
        agent['x'], agent['y'] = new_x, new_y
        self.agent_positions.add((new_x, new_y))

        if agent['destination'] and (new_x, new_y) == agent['destination']:
            agent['destination'] = None

    def step(self):
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
            self.broadcast_message(agent, self.timestep)

        for agent in self.agents:
            agent['messages'] = deque(
                [msg for msg in agent['messages'] if self.timestep - msg['timestep'] <= self.params['message_expiry']],
                maxlen=100)

        alive_agents = []
        for agent in self.agents:
            if agent['sugar'] <= 0:
                self.dead_agents.append({'x': agent['x'], 'y': agent['y'], 'death_time': self.timestep})
                self.agent_positions.remove((agent['x'], agent['y']))
            else:
                alive_agents.append(agent)
        self.agents = alive_agents

        self.dead_agents = [agent for agent in self.dead_agents if self.timestep - agent['death_time'] <= 5]

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

    def run_simulation(self, max_timesteps=1000):
        running = True
        while running and self.timestep < max_timesteps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.step()
            self.render()
            self.clock.tick(5)

        self.plot_results()

# Run the simulation
env = SugarscapeEnvironment(width=50, height=50, num_agents=1000, cell_size=10,
                            broadcast_radius=15,
                            show_sugar_levels=False,
                            show_broadcast_radius=False,
                            show_agent_paths=True)
env.run_simulation()