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
        self.target_lane_center_y = None

    def make_decision(self, obs):
        self.steps += 1
        car_lane = obs[0]
        agent_lane = obs[3]
        car_speed = obs[1]
        agent_speed = obs[4]
        agent_heading = obs[5]
        dx = obs[6]
        agent_y = obs[9]
        lane_2_error = obs[10]
        lane_3_error = obs[11]


        if self.real_init:
            # Process obs and get action from actor
            action = self.actor(obs)
            turn_cmd, acc_cmd = action[0], action[1]
            return turn_cmd, acc_cmd
        else:
            turn_cmd = 0.0
            acc_cmd = 0.0
            # detect for lane change every 20 steps
            if (not self.lane_change_active) and (car_lane != agent_lane) and (self.steps % 20 == 0):
                self.lane_change_active = True

                # Choose lane 2 or lane 3
                if agent_lane == 2:
                    self.target_lane = 3
                else:
                    self.target_lane = 2

            # move to the chosen lane
            if self.lane_change_active:
                if self.target_lane == 2:
                    target_lane_error = lane_2_error
                else:
                    target_lane_error = lane_3_error

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
            if car_lane == agent_lane:
                if dx > 0: # agent ahead
                    target_speed = car_speed - 30.0
                else: # agent behind
                    target_speed = car_speed + 10.0

                target_speed = np.clip(target_speed, configs.MIN_SPEED, configs.MAX_SPEED)
                speed_diff = target_speed - agent_speed

                if speed_diff > 1.0:
                    acc_cmd = configs.MAX_ACC
                elif speed_diff < -1.0:
                    acc_cmd = -configs.MAX_ACC
                else:
                    acc_cmd = 0.0

                if agent_heading > 0.03:
                    turn_cmd = -0.02
                elif agent_heading < -0.03:
                    turn_cmd = 0.02
                else:
                    turn_cmd = 0.0

            return turn_cmd, acc_cmd