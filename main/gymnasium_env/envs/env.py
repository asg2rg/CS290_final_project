import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

dt = 0.2
car_velocity = 25
agent_1_velocity = 50

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

        # car info
        self.car = np.array([100.0, 100.0, 0.0], dtype=np.float32)
        self.agent_1 = np.array([440.0, 400.0, np.pi], dtype=np.float32)
        

        # speeds
        self.car_speed = car_velocity
        self.agent_1_speed = agent_1_velocity
        self.max_speed = 50.0
        self.min_speed = 0.0
        self.omega = 1.0

        # observation:
        low = np.array([0, 0.0, -np.pi, 0, 0.0, -np.pi, 0.0, -np.pi], dtype=np.float32)
        high = np.array([self.num_roads - 1, self.max_speed, np.pi, self.num_roads - 1, self.max_speed, np.pi, np.inf, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # actions:
        # 0 forward
        # 1 turn left
        # 2 turn right
        # 3 accelerate
        # 4 decelerate
        # 5 hard accelerate
        # 6 hard brake
        self.action_space = spaces.Discrete(7)

    #helper function to get the y coordinate of the center of a road given its id
    def road_center_y(self, road_id):
        return self.road_top + (road_id + 0.5) * self.road_width

    def y_to_road_id(self, y):
        road_id = int((y - self.road_top) // self.road_width)
        return int(np.clip(road_id, 0, self.num_roads - 1))

    def _get_obs(self):
        car_road_id = self.y_to_road_id(self.car[1])
        agent_1_road_id = self.y_to_road_id(self.agent_1[1])

        dx = self.agent_1[0] - self.car[0]
        dy = self.agent_1[1] - self.car[1]
        rel_dist = np.sqrt(dx**2+dy**2)
        rel_angle = np.arctan2(dy, dx)
        heading_error = rel_angle - self.car[2]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        return np.array([car_road_id, self.car_speed, self.car[2], agent_1_road_id, self.agent_1_speed, self.agent_1[2], rel_dist, heading_error], dtype=np.float32)

    def _get_info(self):
        return {
            "car_x": self.car[0],
            "car_road": self.y_to_road_id(self.car[1]),
            "agent_1_x": self.agent_1[0],
            "agent_1_road": self.y_to_road_id(self.agent_1[1])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.car[0] = 100.0
        self.car[1] = self.road_center_y(2)
        self.car[2] = 0.0
        self.car_speed = car_velocity

        self.agent_1[0] = 940.0
        self.agent_1[1] = self.road_center_y(1)
        self.agent_1[2] = np.pi
        self.agent_1_speed = agent_1_velocity

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1

        alpha = self.car[2]
        speed = self.car_speed   # base speed for this step

        if action == 1: # turn left 
            alpha -= dt * self.omega
            speed = self.car_speed * 0.6
        elif action == 2: # turn right visually
            alpha += dt * self.omega
            speed = self.car_speed * 0.6 
        elif action == 3: # accelerate
            self.car_speed = min(self.max_speed, self.car_speed + 2.0)
            speed = self.car_speed
        elif action == 4: # decelerate
            self.car_speed = max(self.min_speed, self.car_speed - 2.0)
            speed = self.car_speed
        elif action == 5: # hard accelerate
            self.car_speed = min(self.max_speed, self.car_speed + 10.0)
            speed = self.car_speed
        elif action == 6: # hard brake
            self.car_speed = max(self.min_speed, self.car_speed - 10.0)
            speed = self.car_speed

        if alpha > np.pi:
            alpha -= 2*np.pi
        if alpha < -np.pi:
            alpha += 2*np.pi
        
        # move forward
        self.car[0] += dt * speed * np.cos(alpha)
        self.car[1] += dt * speed * np.sin(alpha)
        self.car[2] = alpha

        # update agent_1 position
        self.agent_1[0] += self.agent_1_speed * dt * np.cos(self.agent_1[2])
        self.agent_1[1] += self.agent_1_speed * dt * np.sin(self.agent_1[2])
        self.respawn_agent_if_offscreen()
        
        truncated = self.step_count >= self.max_episode_steps
        terminated = False # for now

        # placeholder reward for now
        reward = 0.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def respawn_agent_if_offscreen(self):
        agent_screen_x = self.world_to_screen_x(self.agent_1[0])

        if agent_screen_x < -self.car_length:
            self.agent_1[0] = self.car[0] + (self.window_width - self.camera_x) + 100
            self.agent_1[1] = self.road_center_y(1)
            self.agent_1[2] = np.pi
            self.agent_1_speed = agent_1_velocity
            
    # AI generated for rendering code
    def world_to_screen_x(self, world_x):
        return int(world_x - self.car[0] + self.camera_x)

    def draw_vehicle(self, canvas, x, y, heading, color):
        screen_x = self.world_to_screen_x(x)
        screen_y = int(y)

        # Create a small car surface
        car_surface = pygame.Surface((self.car_length, self.car_width), pygame.SRCALPHA)

        # Car body
        pygame.draw.rect(
            car_surface,
            color,
            pygame.Rect(0, 0, self.car_length, self.car_width),
            border_radius=6
        )

        # Windshield near the front of the car
        windshield_rect = pygame.Rect(
            self.car_length - 16,   # near the front
            4,                      # a little down from the top
            10,                     # windshield width
            self.car_width - 8      # windshield height
        )
        pygame.draw.rect(
            car_surface,
            (180, 220, 255),        # light blue
            windshield_rect,
            border_radius=3
        )

        pygame.draw.rect(
            car_surface,
            (120, 160, 200),
            windshield_rect,
            width=1,
            border_radius=3
        )

        # Rotate according to heading
        angle_degrees = -np.degrees(heading)
        rotated_surface = pygame.transform.rotate(car_surface, angle_degrees)
        rotated_rect = rotated_surface.get_rect(center=(screen_x, screen_y))

        canvas.blit(rotated_surface, rotated_rect)

    def draw_road(self, canvas):
        road_rect = pygame.Rect(0, self.road_top, self.window_width, self.num_roads * self.road_width)
        pygame.draw.rect(canvas, (70, 70, 70), road_rect)

        # solid road boundary / lane lines
        for i in range(self.num_roads + 1):
            y = self.road_top + i * self.road_width

            # center divider between opposite directions
            if i == self.num_roads // 2:
                pygame.draw.line(canvas, (255, 220, 0), (0, y), (self.window_width, y), 4)
            else:
                pygame.draw.line(canvas, (255, 255, 255), (0, y), (self.window_width, y), 2)

        # dashed line on the road center
        dash_spacing = 80
        dash_length = 30

        world_left = self.car[0] - self.camera_x
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
        self.draw_vehicle(canvas, self.car[0], self.car[1], self.car[2], (50, 150, 255))
        self.draw_vehicle(canvas, self.agent_1[0], self.agent_1[1], self.agent_1[2], (20, 20, 20))
    
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
            self.window = None
            self.clock = None