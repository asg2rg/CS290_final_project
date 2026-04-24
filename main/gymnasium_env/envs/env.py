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
        self.collision_margin = 1.5

        # car info
        self.car = np.array([100.0, 100.0, 0.0], dtype=np.float32)
        self.agent_1 = np.array([440.0, 400.0, np.pi], dtype=np.float32)

        # speeds
        self.car_speed = car_velocity
        self.agent_1_speed = agent_1_velocity
        self.max_speed = 100.0
        self.min_speed = 0.0
        self.omega = 1.0

        # observation:
        low = np.array([-1, 0.0, -np.pi, -1, 0.0, -np.pi, 0.0, -np.pi], dtype=np.float32)
        high = np.array([self.num_roads - 1, self.max_speed, np.pi, self.num_roads - 1, self.max_speed, np.pi, np.inf, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # turn_cmd:
        # 0 slight left: -1.0
        # 1 med left: -2.0
        # 2 hard left: -4.0
        # 3 no turn
        # 4 slight right: 1.0
        # 5 med right: 2.0
        # 6 hard right: 4.0

        # acc_cmd:
        # 0 slight accel: 2.0
        # 1 med accel: 5.0
        # 2 hard accel: 10.0
        # 3 no accel: 0.0
        # 4 slight brake: -2.0
        # 5 med brake: -5.0
        # 6 hard brake: -10.0
        self.action_space = spaces.MultiDiscrete([7, 7]) # turn_cmd, acc_cmd

    #helper function to get the y coordinate of the center of a road given its id
    def road_center_y(self, road_id):
        return self.road_top + (road_id + 0.5) * self.road_width

    def y_to_road_id(self, y):
        if y < self.road_top or y > self.road_bottom:
            return -1
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

    def termin_check(self):
        # collision
        if self.collision_check():
            print("Collision")
            return True
        # terminate when the car is >20 units into the green area
        if self.car[1] < self.road_top - 20.0:
            print("Too high")
            return True
        if self.car[1] > self.road_bottom + 20.0:
            print("Too low")
            return True
        return False
    
    def collision_check(self):
        x_zone = (self.car_length/2) + self.collision_margin
        y_zone = (self.car_width/2) + self.collision_margin
        # simple collision check
        car_x = self.car[0]
        car_y = self.car[1]
        car_yaw = self.car[2]
        car_corners = np.array([
            self.rotate_point(x_zone, -y_zone, car_yaw),
            self.rotate_point(x_zone, y_zone, car_yaw),
            self.rotate_point(-x_zone, y_zone, car_yaw),
            self.rotate_point(-x_zone, -y_zone, car_yaw)
        ]) + np.array([car_x, car_y])

        agent_1_x = self.agent_1[0]
        agent_1_y = self.agent_1[1]
        agent_1_yaw = self.agent_1[2]
        agent_1_corners = np.array([
            self.rotate_point(x_zone, -y_zone, agent_1_yaw),
            self.rotate_point(x_zone, y_zone, agent_1_yaw),
            self.rotate_point(-x_zone, y_zone, agent_1_yaw),
            self.rotate_point(-x_zone, -y_zone, agent_1_yaw)
        ]) + np.array([agent_1_x, agent_1_y])

        return self.point_in_rectangle(car_corners, agent_1_corners) or self.point_in_rectangle(agent_1_corners, car_corners)

    #### Copilot-generated separating axis theorem check ####
    def rectangle_axes(self, corners):
        axes = []
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]

            # A separating axis is the outward normal of an edge.
            # For a rectangle, testing the 4 edge normals is enough.
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]], dtype=np.float32)
            norm = np.linalg.norm(axis)
            if norm > 0:
                axes.append(axis / norm)

        return axes

    def project_onto_axis(self, corners, axis):
        # Projection collapses the polygon onto a 1D line.
        # If the 1D intervals do not overlap on any axis, the rectangles do not collide.
        projections = np.dot(corners, axis)
        return np.min(projections), np.max(projections)

    def point_in_rectangle(self, corners1, corners2):
        # corners in clockwise order, check rotated rectangle collision
        for axis in self.rectangle_axes(corners1) + self.rectangle_axes(corners2):
            min1, max1 = self.project_onto_axis(corners1, axis)
            min2, max2 = self.project_onto_axis(corners2, axis)

            # A gap on any separating axis means the rectangles do not overlap.
            if max1 < min2 or max2 < min1:
                return False
        return True
    #### I hate math ####
    
    def rotate_point(self, x, y, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        return x_rot, y_rot
    
    def parse_cmds(self, turn, acc):
        turn_cmds = {
            0: -1.0, # slight left
            1: -2.0, # med left
            2: -3.0, # hard left
            3: 0.0,  # no turn
            4: 1.0,  # slight right
            5: 2.0,  # med right
            6: 3.0   # hard right
        }
        acc_cmds = {
            0: 2.0,   # slight accel
            1: 5.0,   # med accel
            2: 10.0,  # hard accel
            3: 0.0,   # no accel
            4: -2.0,  # slight brake
            5: -5.0,  # med brake
            6: -10.0   # hard brake
        }
        return turn_cmds[turn], acc_cmds[acc] # omega, spd

    def step(self, action):
        self.step_count += 1

        alpha = self.car[2]
        speed = self.car_speed   # base speed for this step

        turn_cmd = action[0]
        acc_cmd = action[1]

        turn_delta, acc_delta = self.parse_cmds(turn_cmd, acc_cmd)
        alpha += dt * turn_delta * self.omega
        spd = np.clip(speed + acc_delta, self.min_speed, self.max_speed)
        self.car_speed = spd

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
        terminated = self.termin_check()

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