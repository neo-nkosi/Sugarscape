import random
import numpy as np
import pygame
import matplotlib.pyplot as plt


class SugarscapeEnvironment:
    def __init__(self, width, height, num_agents, cell_size=15, show_sugar_levels=True):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.cell_size = cell_size
        self.show_sugar_levels = show_sugar_levels

        self.params = {
            'max_sugar': 5,
            'growth_rate': 1,
            'sugar_peak_frequency': 0.04,
            'sugar_peak_spread': 6,
            'job_center_duration': (40, 100)
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
        pygame.display.set_caption("Sugarscape Simulation - Limited Vision")
        self.clock = pygame.time.Clock()

        # Initialize font for rendering text
        self.font = pygame.font.Font(None, 10)  # You may need to adjust the font size

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
        # Round sugar levels to nearest integer
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
            'metabolism': np.random.randint(1,3),
        }

    def get_adjacent_cells(self, x, y):
        adjacent = [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1),
        ]
        return [(x, y) for x, y in adjacent if 0 <= x < self.width and 0 <= y < self.height]

    def move_agent(self, agent):
        x, y = agent['x'], agent['y']
        adjacent_cells = self.get_adjacent_cells(x, y)

        best_cell = None
        best_sugar = -1

        for cell_x, cell_y in adjacent_cells:
            if (cell_x, cell_y) not in self.agent_positions:
                sugar_amount = self.sugar[cell_y, cell_x]
                if sugar_amount > best_sugar:
                    best_sugar = sugar_amount
                    best_cell = (cell_x, cell_y)

        if best_cell and best_sugar > 0:
            self.agent_positions.remove((agent['x'], agent['y']))
            agent['x'], agent['y'] = best_cell
            self.agent_positions.add((agent['x'], agent['y']))
        else:
            # Move randomly if no sugar is visible
            available_cells = [cell for cell in adjacent_cells if cell not in self.agent_positions]
            if available_cells:
                new_x, new_y = random.choice(available_cells)
                self.agent_positions.remove((agent['x'], agent['y']))
                agent['x'], agent['y'] = new_x, new_y
                self.agent_positions.add((agent['x'], agent['y']))
            # If no available cells, the agent stays put

    def step(self):
        # Update job centers
        for center in self.job_centers:
            center['duration'] -= 1
        self.job_centers = [center for center in self.job_centers if center['duration'] > 0]
        if np.random.random() < self.params['sugar_peak_frequency']:
            self.create_job_center()
        self.update_sugar_landscape()

        # Move agents
        for agent in self.agents:
            self.move_agent(agent)

        # Collect sugar and apply metabolism
        for agent in self.agents:
            collected_sugar = self.sugar[agent['y'], agent['x']]
            agent['sugar'] += collected_sugar
            self.sugar[agent['y'], agent['x']] = 0
            agent['sugar'] -= agent['metabolism']
            agent['sugar'] = int(agent['sugar'])  # Ensure agent sugar is also an integer

        # Handle agent death
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

    def render(self):
        self.screen.fill((255, 255, 255))

        for y in range(self.height):
            for x in range(self.width):
                sugar_level = self.sugar[y, x]
                color = self.get_color(sugar_level)
                pygame.draw.rect(self.screen, color,
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

                # Render sugar quantity as text if show_sugar_levels is True
                if self.show_sugar_levels:
                    sugar_text = self.font.render(f"{sugar_level}", True, (0, 0, 0))
                    text_rect = sugar_text.get_rect(center=(x * self.cell_size + self.cell_size // 2,
                                                            y * self.cell_size + self.cell_size // 2))
                    self.screen.blit(sugar_text, text_rect)

        for agent in self.agents:
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(agent['x'] * self.cell_size + self.cell_size / 2),
                                int(agent['y'] * self.cell_size + self.cell_size / 2)),
                               int(self.cell_size / 3))

        pygame.display.flip()

    def get_color(self, sugar_level):
        if sugar_level == 0:
            return (255, 255, 255)
        else:
            intensity = sugar_level / self.params['max_sugar']
            return (255, 255, int(255 * (1 - intensity)))

    def run_simulation(self, max_timesteps=1000):
        running = True
        while running and self.timestep < max_timesteps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.step()
            self.render()
            self.clock.tick(10)

        self.plot_results()

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


env = SugarscapeEnvironment(width=50, height=50, num_agents=1000, cell_size=10, show_sugar_levels=False)
env.run_simulation(max_timesteps=1000)


