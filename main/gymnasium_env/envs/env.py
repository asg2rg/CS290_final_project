# Observation Space: road_id of the car(0 - 3), car_speed, agent_1_agent_road_id (0 - 3), relative_distance, relative_heading
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

dt = 0.2

class CarAndTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_episode_steps=200):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # initialization
        # define the size of the screen
        self.window_width = 1200
        self.window_height = 500
        self.window = None
        self.clock = None

        # define the road
        self.num_roads = 4
        self.road_width = 80
        self.road_length = 5000
        self.road_top = 90
        self.road_bottom = self.road_top + self.num_roads * self.road_width

        # keep camera on the car and near left-center 
        self.camera_x = 250

        # car geometry
        self.car_length = 40
        self.car_width = 24

        # speeds
        self.car_speed = 20.0
        self.agent_1_speed = 16.0
        self.max_speed = 30.0
        self.min_speed = 0.0

        self.car_x = 0.0
        self.car_road = 1

        self.agent_1_x = 0.0
        self.agent_1_road = 2

        # observation:
        low = np.array([0, 0.0, 0, -np.inf], dtype=np.float32)
        high = np.array([self.num_roads - 1, self.max_speed, self.num_roads - 1, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # actions:
        # 0 accelerate
        # 1 decelerate
        self.action_space = spaces.Discrete(2)

    #helper function to get the y coordinate of the center of a road given its id
    def road_center_y(self, road_id):
        return self.road_top + (road_id + 0.5) * self.road_width

    def _get_obs(self):
        rel_x = self.agent_1_x - self.car_x
        rel_speed = self.agent_1_speed - self.car_speed
        return np.array([
            float(self.car_road),
            float(self.car_speed),
            float(self.agent_1_road),
            float(rel_x),
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "car_x": self.car_x,
            "car_road": self.car_road,
            "agent_1_x": self.agent_1_x,
            "agent_1_road": self.agent_1_road
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.car_x = 100.0
        self.car_road = 1
        self.car_speed = 50.0

        self.agent_1_x = 140.0
        self.agent_1_road = 2
        self.agent_1_speed = 52.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1

        # car action
        if action == 0: # accelerate
            self.car_speed = min(self.max_speed, self.car_speed + 2.0)
        elif action == 1: # decelerate
            self.car_speed = max(self.min_speed, self.car_speed - 2.0)

        # move vehicles forward
        self.car_x += self.car_speed * dt
        self.agent_1_x += self.agent_1_speed * dt

        # simple termination/truncation
        truncated = self.step_count >= self.max_episode_steps
        terminated = False

        # placeholder reward for now
        reward = 0.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # AI generated for rendering code
    def world_to_screen_x(self, world_x):
        return int(world_x - self.car_x + self.camera_x)

    def draw_vehicle(self, canvas, x, road_id, color):
        screen_x = self.world_to_screen_x(x)
        screen_y = int(self.road_center_y(road_id))

        rect = pygame.Rect(
            screen_x - self.car_length // 2,
            screen_y - self.car_width // 2,
            self.car_length,
            self.car_width
        )
        pygame.draw.rect(canvas, color, rect, border_radius=6)

    def draw_road(self, canvas):
        road_rect = pygame.Rect(0, self.road_top, self.window_width, self.num_roads * self.road_width)
        pygame.draw.rect(canvas, (70, 70, 70), road_rect)

        # solid road boundary lines
        for i in range(self.num_roads + 1):
            y = self.road_top + i * self.road_width
            pygame.draw.line(canvas, (255, 255, 255), (0, y), (self.window_width, y), 2)

        # dashed line on the road center
        dash_spacing = 80
        dash_length = 30

        world_left = self.car_x - self.camera_x
        world_right = world_left + self.window_width

        start_dash = int(np.floor(world_left / dash_spacing) * dash_spacing)
        end_dash = int(world_right + dash_spacing)

        for road_id in range(self.num_roads):
            dash_y = int(self.road_center_y(road_id))
            for wx in range(start_dash, end_dash, dash_spacing):
                sx = self.world_to_screen_x(wx)
                pygame.draw.line(canvas, (220, 220, 0), (sx, dash_y), (sx + dash_length, dash_y), 3)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((30, 120, 30))  # outer background

        self.draw_road(canvas)

        # car and agent_1 car
        self.draw_vehicle(canvas, self.car_x, self.car_road, (50, 150, 255))
        self.draw_vehicle(canvas, self.agent_1_x, self.agent_1_road, (20, 20, 20))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()