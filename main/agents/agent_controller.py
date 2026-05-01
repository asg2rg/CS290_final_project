import torch
import numpy as np
import utils.configs as configs
from agents.actor import Actor
from agents.critic import Critic

class AgentController:
    def __init__(self, real_init = False):
        if real_init:
            self.actor = Actor(...)
            self.critic = Critic(...)
        else:
            print("Initialize placeholder Controller")
        self.real_init = real_init
        self.steps = 0
        self.target_lane = None
        self.lane_change_active = False
        self.speed_change_active = False
        self.target_lane_center_y = None

    def make_decision(self, obs):
        self.steps += 1
        car_lane = int(obs[0])
        agent_lane = int(obs[3])
        car_speed = obs[1]
        agent_speed = obs[4]
        agent_heading = obs[5]
        dx = obs[6]
        lane_error = obs[9:13]
        target_lane_error = None

        if self.real_init:
            # Process obs and get action from actor
            action = self.actor(obs)
            turn_cmd, acc_cmd = action[0], action[1]
            return turn_cmd, acc_cmd
        else:
            turn_cmd = 0.0
            acc_cmd = 0.0
            # detect for lane change every 100 steps
            if (not self.lane_change_active) and (self.steps % 100 == 0):
                if (car_lane != agent_lane) or np.random.rand() < 0.3:
                    self.lane_change_active = True
                    self.target_lane = 3 if agent_lane == 2 else 2
            # move to the chosen lane
            if self.lane_change_active:
                # print(f"Changing lane from {agent_lane} to {self.target_lane}")
                target_lane_error = lane_error[self.target_lane]

                lane_tolerance = 5.0
                heading_tolerance = 0.03

                desired_heading = 0.003 * target_lane_error
                desired_heading = np.clip(desired_heading, -0.15, 0.15)

                if agent_heading < desired_heading - heading_tolerance:
                    turn_cmd = 0.07
                elif agent_heading > desired_heading + heading_tolerance:
                    turn_cmd = -0.07
                else:
                    turn_cmd = 0.0

                target_speed = car_speed
                speed_diff = target_speed - agent_speed

                if speed_diff > 1.0:
                    acc_cmd = configs.MAX_ACC
                elif speed_diff < -1.0:
                    acc_cmd = -configs.MAX_ACC
                else:
                    acc_cmd = 0.0

                if (agent_lane == self.target_lane and abs(target_lane_error) < lane_tolerance and abs(agent_heading) < heading_tolerance):
                    self.lane_change_active = False
                    self.target_lane = None

                return turn_cmd, acc_cmd

            # if in the same lane
            # if car_lane == agent_lane:
            if not self.speed_change_active and self.steps % 70 == 0:
                spd_probs = np.random.rand()
                if spd_probs < 0.3 or (car_lane == agent_lane and spd_probs < 0.9):
                # might stay same even if in same lane, small chance to change even if different lane
                    irrational_probs = np.random.rand()
                    if dx > 0: # agent ahead
                        if irrational_probs < 0.1:
                            target_speed = configs.TARGET_SPEED + 20.0 # speed up even if ahead
                        else:
                            target_speed = configs.TARGET_SPEED - 20.0
                    else: # agent behind
                        if irrational_probs < 0.1:
                            target_speed = configs.TARGET_SPEED - 20.0 # slow dodwn even if behind
                        else:
                            target_speed = configs.TARGET_SPEED + 10.0
                    self.speed_change_active = True
            if self.speed_change_active:
                target_speed = np.clip(target_speed, configs.MIN_SPEED, configs.MAX_SPEED)
                speed_diff = target_speed - agent_speed

                if speed_diff > configs.MAX_ACC/1.5:
                    acc_cmd = configs.MAX_ACC
                    # print(f"Accelerating to {target_speed:.1f}")
                elif speed_diff < -configs.MAX_ACC/1.5:
                    acc_cmd = -configs.MAX_ACC
                    # print(f"Braking to {target_speed:.1f}")
                else:
                    acc_cmd = 0.0
                    self.speed_change_active = False

                if agent_heading > 0.03:
                    turn_cmd = -0.02
                elif agent_heading < -0.03:
                    turn_cmd = 0.02
                else:
                    turn_cmd = 0.0

            return turn_cmd, acc_cmd