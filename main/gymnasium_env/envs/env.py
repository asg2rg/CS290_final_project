import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from agents.agent_car import AgentCar

import utils.configs as configs

dt = configs.TIMESTEP
car_velocity = configs.CAR_INITIAL_VEL
agent_1_velocity = configs.AGENT_1_INITIAL_VEL

MIN_RWD = -500.0

class CarAndTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # initialization
        # define the size of the screen
        self.window_width = configs.WINDOW_W
        self.window_height = configs.WINDOW_H
        self.window = None
        self.clock = None
        self.font = None

        # define the road
        self.num_roads = configs.ROAD_CNT
        self.road_width = configs.ROAD_W
        self.road_length = configs.ROAD_L
        self.road_top = configs.ROAD_TOP
        self.road_bottom = self.road_top + self.num_roads * self.road_width

        # keep camera on the car and near left-center 
        self.camera_x = configs.CAM_X

        # car geometry
        self.car_length = configs.CAR_L
        self.car_width = configs.CAR_W
        self.collision_margin = 1.5

        # car info
        self.car = np.array([100.0, 100.0, 0.0], dtype=np.float32) # car init x at 100
        self.agents = [AgentCar(id = i, x=350.0 + i*100, y=self.road_center_y(i), heading=np.pi, speed=agent_1_velocity) for i in range(configs.AGENT_CNT)]
        # self.agents[0].heading *= -1
        # self.agents[1].heading *= -1
        
        self.last_x = 100.0
        self.last_lane = 2

        # speeds
        self.car_speed = car_velocity
        self.max_speed = configs.MAX_SPEED
        self.min_speed = configs.MIN_SPEED
        self.omega = configs.TURN_UNIT

        # observation:
        low = np.array(
            [-1.0, 0.0, -np.pi] + [
                -1.0, 0.0, -np.pi, 0.0, -np.pi, 0.0
            ] * configs.NEAREST_AGENTS,
            dtype=np.float32,
        )
        high = np.array(
            [self.num_roads - 1, self.max_speed, np.pi] + [
                self.num_roads - 1, self.max_speed, np.pi, np.inf, np.pi, 1.0
            ] * configs.NEAREST_AGENTS,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if configs.DISCRETE:
            self.action_space = spaces.MultiDiscrete([configs.TURN_SHAPE, configs.ACC_SHAPE]) # turn_cmd, acc_cmd
        else:
            self.action_space = spaces.Box(low=np.array([-configs.MAX_ANG, -configs.MAX_ACC]), high=np.array([configs.MAX_ANG, configs.MAX_ACC]), shape=(2,), dtype=np.float32) # turn_cmd, acc_cmd
    #helper function to get the y coordinate of the center of a road given its id
    def road_center_y(self, road_id):
        return self.road_top + (road_id + 0.5) * self.road_width

    def y_to_road_id(self, y):
        if y < self.road_top or y > self.road_bottom:
            return -1
        road_id = int((y - self.road_top) // self.road_width)
        return int(np.clip(road_id, 0, self.num_roads - 1))

    def _build_agent_obs(self, target_agent, reference_entity=None):
        if reference_entity is None:
            reference_entity = self.car
        
        agent_road_id = self.y_to_road_id(target_agent.state[1])
        dx = target_agent.state[0] - reference_entity[0]
        dy = target_agent.state[1] - reference_entity[1]
        rel_dist = np.sqrt(dx**2 + dy**2)
        rel_angle = np.arctan2(dy, dx)
        heading_error = rel_angle - target_agent.state[2]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        return [1.0, agent_road_id, target_agent.speed, target_agent.state[2], rel_dist, heading_error]

    def _update_agent_side_counts(self):
        configs.AGENTS_FRONT = sum(1 for agent in self.agents if agent.state[0] > self.car[0])
        configs.AGENTS_BEHIND = sum(1 for agent in self.agents if agent.state[0] < self.car[0])

    def _get_reference_agent(self):
        if not self.agents:
            return None
        return min(self.agents, key=lambda agent: (agent.state[0] - self.car[0]) ** 2 + (agent.state[1] - self.car[1]) ** 2)

    # observation space for the car
    def _get_obs(self):
        car_road_id = self.y_to_road_id(self.car[1])
        obs_parts = [car_road_id, self.car_speed, self.car[2]]

        if not self.agents:
            nearest_indices = []
        else:
            agent_distances = []
            for i, agent in enumerate(self.agents):
                dx = agent.state[0] - self.car[0]
                dy = agent.state[1] - self.car[1]
                dist = np.sqrt(dx**2 + dy**2)
                agent_distances.append((dist, i))
            agent_distances.sort()
            nearest_indices = [i for _, i in agent_distances[:configs.NEAREST_AGENTS]]

        for nearest_idx in nearest_indices:
            obs_parts.extend(self._build_agent_obs(self.agents[nearest_idx]))

        for _ in range(configs.NEAREST_AGENTS - len(nearest_indices)):
            obs_parts.extend([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs_parts, dtype=np.float32)

    def _get_nearest_agents(self, agent_idx, k=4):
        """Get indices of k nearest agents to the given agent (excluding itself)"""
        agent = self.agents[agent_idx]
        distances = []
        for i, other in enumerate(self.agents):
            if i != agent_idx:
                dx = other.state[0] - agent.state[0]
                dy = other.state[1] - agent.state[1]
                dist = np.sqrt(dx**2 + dy**2)
                distances.append((dist, i))
        distances.sort()
        return [i for _, i in distances[:k]] + [-1] * (k - len(distances))

    def _get_agent_obs(self, agent_idx):
        """Get observation for agent_idx from that agent's POV: agent's own state + 4 nearest other agents"""
        agent = self.agents[agent_idx]
        agent_road_id = self.y_to_road_id(agent.state[1])
        obs_parts = [agent_road_id, agent.speed, agent.state[2]]

        # get car info
        car_lane = self.y_to_road_id(self.car[1])
        car_speed = self.car_speed
        car_yaw = self.car[2]
        dx = agent.state[0] - self.car[0]
        dy = agent.state[1] - self.car[1]
        lane_err = [self.road_center_y(i) - agent.state[1] for i in range(configs.ROAD_CNT)]
        obs_parts.extend([car_lane, car_speed, car_yaw, dx, dy] + lane_err)

        nearest_indices = self._get_nearest_agents(agent_idx, configs.NEAREST_AGENTS-1)
        for nearest_idx in nearest_indices:
            if nearest_idx >= 0:
                nearest_agent = self.agents[nearest_idx]
                obs_parts.extend(self._build_agent_obs(nearest_agent, reference_entity=agent.state))
            else:
                obs_parts.extend([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        # print(f"Agent {agent_idx} obs: {obs_parts}")
        
        return np.array(obs_parts, dtype=np.float32)


    def _get_info(self):
        info = {
            "car_x": self.car[0],
            "car_road": self.y_to_road_id(self.car[1]),
            "agent_count": len(self.agents),
            "agents_front": configs.AGENTS_FRONT,
            "agents_behind": configs.AGENTS_BEHIND
        }
        for i, agent in enumerate(self.agents):
            info[f"agent_{i}_x"] = agent.state[0]
            info[f"agent_{i}_road"] = self.y_to_road_id(agent.state[1])
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.eps_reward = 0.0

        self.car[0] = 100.0
        self.last_x = 100.0
        self.car[1] = self.road_center_y(2)
        self.car[2] = 0.0
        self.last_lane = self.y_to_road_id(self.car[1])
        configs.TARGET_LANE = np.random.choice([2, 3])
        self.car_speed = car_velocity

        for i, agent in enumerate(self.agents):
            lane = np.random.choice(4)
            heading = 0
            if lane in [0, 1]:
                heading = np.pi
            x_offset = (np.random.random() - 0.5) * 200
            spd = agent_1_velocity + (np.random.random() - 0.5) * 10
            agent.reset(x=400 + (i%4)*200 + x_offset, y=self.road_center_y(lane), heading=heading, speed=spd)
        self._update_agent_side_counts()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def reward_calc(self, turn_cmd, acc_cmd):
        reward = -0.1 # timestep penalty
        yaw = abs(self.car[2])
        yaw_deg = np.degrees(yaw)
        speed_diff = abs(self.car_speed - configs.TARGET_SPEED)
        #### PENALTIES ####
        # collision penalty
        if self.collision_check():
            col_rwd = -10.0
            reward += col_rwd
        # OOB penalty
        if self.boundary_check():
            oob_rwd = -3.0
            reward += oob_rwd
        # large penalty for yaw > 70 degrees
        if yaw_deg > 90.0:
            yaw_rwd = -5.0
            reward += yaw_rwd
            # print(f"Reversed penalty: {yaw_rwd}")
        elif yaw_deg > 60.0:
            yaw_rwd = -3.0
            reward += yaw_rwd
            # print(f"Large yaw penalty: {yaw_rwd}")
        elif yaw_deg > 30.0:
            yaw_rwd = -1.0
            reward += yaw_rwd
        elif yaw_deg > 10.0:
            yaw_rwd = -0.5
            reward += yaw_rwd
        if not configs.DISCRETE:
            # penalize excessive speed
            if abs(acc_cmd) > configs.MAX_ACC:
                reward -= (abs(acc_cmd) - configs.MAX_ACC) * 0.1
                # print(f"Excessive acceleration penalty: acc {acc_cmd}")
            # penalize excessive turning
            if abs(turn_cmd) > configs.MAX_ANG:
                reward -= (abs(turn_cmd) - configs.MAX_ANG) * 0.3
                # print(f"Excessive turning penalty: turn {turn_cmd}")
            if speed_diff > 30.0:
                spd_rwd = (speed_diff - 5.0) * -0.1
                reward += spd_rwd
                # print(f"Diff speed penalty: {spd_rwd}")
        
        #### REWARDS ####
        # only if speed > 0
        if self.car_speed < 0.1:
            return reward
        # positive distance reward
        dist_rwd = self.car[0] - self.last_x
        reward += min(dist_rwd * 0.01, 0.07)
        # reward close to target speed and straight
        if yaw_deg < 3.0:
            reward += 0.5
        elif yaw_deg < 7.0:
            reward += 0.2
        speed_rwd = max(1.0 - speed_diff * 0.1, 0.0)
        reward += speed_rwd

        # lane control
        lane = self.y_to_road_id(self.car[1])
        # penalize lane switching
        if lane != self.last_lane:
            lane_change_rwd = -0.3
            reward += lane_change_rwd
            # print(f"Lane change penalty: {lane_change_rwd}")
            self.last_lane = lane
        # reward being in right lane
        if lane == configs.TARGET_LANE:
            lane_rwd = 3.0
            reward += lane_rwd
        elif (configs.TARGET_LANE in [2, 3] and lane in [2, 3]) or (configs.TARGET_LANE in [0, 1] and lane in [0, 1]):
            reward += 0.5
        else:
            # wrong side of road or off road
            reward += -1.5
        # penalize far from road center
        dist_to_lane_center = abs(self.car[1] - self.road_center_y(lane)) # range 0~40
        dist_center_rwd = 0.3-((dist_to_lane_center**2 / 5) * 0.005)
        reward += dist_center_rwd
        dist_to_tgt_center = abs(self.car[1] - self.road_center_y(configs.TARGET_LANE))
        dist_tgt_center_rwd = 0.3-min((dist_to_tgt_center * 0.005), 0.4)
        reward += dist_tgt_center_rwd
        # print(dist_center_rwd)
            
        #### JERKING PENALTIES ####
        # penalize yaw
        if yaw_deg > 7.0:
            yaw_rwd = yaw_deg * -0.05
            reward += yaw_rwd
            # print(f"Yaw penalty: {yaw_rwd}")
        if abs(turn_cmd) > 1.0:
            turn_rwd = abs(turn_cmd) * -0.1
            reward += turn_rwd
        return reward

    def boundary_check(self):
        if self.car[1] < self.road_top - 20.0:
            # print("Too high")
            return True
        if self.car[1] > self.road_bottom + 20.0:
            # print("Too low")
            return True
        return False

    def termin_check(self, reward):
        # time out handled by truncate
        # < min reward
        if reward < MIN_RWD:
            # print(f"Terminating for low reward: {reward}")
            return True
        if configs.EVAL:
            # terminate if collision
            if self.collision_check():
                # print("Terminating for collision")
                return True
            # terminate if OOB
            if self.boundary_check():
                # print("Terminating for OOB")
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

        for agent in self.agents:
            agent_x = agent.state[0]
            agent_y = agent.state[1]
            agent_yaw = agent.state[2]
            agent_corners = np.array([
                self.rotate_point(x_zone, -y_zone, agent_yaw),
                self.rotate_point(x_zone, y_zone, agent_yaw),
                self.rotate_point(-x_zone, y_zone, agent_yaw),
                self.rotate_point(-x_zone, -y_zone, agent_yaw)
            ]) + np.array([agent_x, agent_y])
            if self.point_in_rectangle(car_corners, agent_corners):
                return True
        return False

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
        return configs.turn_cmds[turn], configs.acc_cmds[acc] # omega, spd

    def step(self, action):
        self.step_count += 1
        # change target lane every 100 steps
        if self.step_count % 180 == 0:
            if configs.TARGET_LANE == 3:
                configs.TARGET_LANE = 2
            else:
                configs.TARGET_LANE = 3
        if self.step_count % 240 == 0:
            configs.TARGET_SPEED = np.random.choice([35, 45, 55, 65, 75]) + np.random.uniform(-5.0, 5.0)
            # print(f"New target lane: {configs.TARGET_LANE}, new target speed: {configs.TARGET_SPEED:.2f}")

        alpha = self.car[2]
        speed = self.car_speed   # base speed for this step

        turn_cmd = action[0]
        acc_cmd = action[1]

        if configs.DISCRETE:
            turn_delta, acc_delta = self.parse_cmds(turn_cmd, acc_cmd)
            clip_acc = acc_delta
            clip_turn = turn_delta
        else:
            turn_delta = turn_cmd
            acc_delta = acc_cmd
            clip_acc = np.clip(acc_delta, -configs.MAX_ACC, configs.MAX_ACC)
            # note: accel value * (1/dt) = accel km/h/s; with dt=0.2 accel hits 50km/h in 1s
            # assume realistic 100km/h in 5s: 20km/h/s, MAX_ACC for 0.2dt should be 4.0
            clip_turn = np.clip(turn_delta, -configs.MAX_ANG, configs.MAX_ANG)
            # clip alpha has been trimmed by dt already
            # in short: our car is a straight line rocket with crappy turns
            
        alpha += dt * clip_turn * self.omega
        spd = np.clip(speed + clip_acc, self.min_speed, self.max_speed)
        self.car_speed = spd

        if alpha > np.pi:
            alpha -= 2*np.pi
        if alpha < -np.pi:
            alpha += 2*np.pi
        
        # move forward
        # if self.collision_check():
        #     speed *= 0.05
        #     alpha *= -0.5
        self.car[0] += dt * speed * np.cos(alpha)
        self.car[1] += dt * speed * np.sin(alpha)
        self.car[2] = alpha

        # print(f"Car position: ({self.car[0]:.2f}, {self.car[1]:.2f}), speed: {self.car_speed:.2f}, heading: {np.degrees(self.car[2]):.2f} degrees")

        # update all agents position
        for i, agent in enumerate(self.agents):
            agent_obs = self._get_agent_obs(i)
            agent.step(dt, agent_obs) # update with internal state
            self.respawn_agent_if_offscreen(i, self.y_to_road_id(self.car[1]))

        self._update_agent_side_counts()

        reward = self.reward_calc(turn_delta, acc_delta)
        truncated = self.step_count >= self.max_episode_steps
        self.eps_reward += reward
        self.last_x = self.car[0]
        terminated = self.termin_check(self.eps_reward)
        done = terminated or truncated
        # if done:
        #     # add final distance reward
        #     dist_traveled = self.car[0] - 100.0
        #     reward += (dist_traveled - 200) * 0.1 # if didn't go far enough, becomes penalty

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def respawn_agent_if_offscreen(self, agent_idx, road_id=1):
        agent = self.agents[agent_idx]
        agent_screen_x = self.world_to_screen_x(agent.state[0])

        agent_idx %= 4
        heading = 0
        if road_id in [0, 1]:
            heading = np.pi
        if agent_screen_x < -self.car_length:
            agent.reset(x=self.car[0] + (self.window_width - self.camera_x) + 100 + agent_idx*50, y=self.road_center_y(road_id), heading=heading, speed=agent_1_velocity)
        agent_screen_x = self.world_to_screen_x(agent.state[0])


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
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("consolas", 22)

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((30, 120, 30))  # outer background

        self.draw_road(canvas)

        # car and agents
        self.draw_vehicle(canvas, self.car[0], self.car[1], self.car[2], (50, 150, 255))
        for agent in self.agents:
            self.draw_vehicle(canvas, agent.state[0], agent.state[1], agent.state[2], (20, 20, 20))

        # HUD in the upper-left corner
        hud_x = 16
        hud_y = 12
        hud_lines = [
            f"SPD: {self.car_speed:.1f}/{configs.TARGET_SPEED:.1f}",
            f"RWD: {self.eps_reward:.1f}",
            f"Lane: {self.y_to_road_id(self.car[1])}/{configs.TARGET_LANE}",
            f"Distance: {self.car[0] - 100.0:.1f}",
        ]
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        line_spacing = 26
        hud_width = 210
        hud_height = 16 + line_spacing * len(hud_lines)
        pygame.draw.rect(canvas, bg_color, pygame.Rect(hud_x - 8, hud_y - 6, hud_width, hud_height), border_radius=6)
        for idx, line in enumerate(hud_lines):
            text_surface = self.font.render(line, True, text_color)
            canvas.blit(text_surface, (hud_x, hud_y + idx * line_spacing))
    
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