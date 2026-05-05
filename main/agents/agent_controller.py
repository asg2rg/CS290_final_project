import torch
import numpy as np
import utils.configs as configs
from agents.actor import Actor
from agents.critic import Critic

class AgentController:
    def __init__(self, real_init = False, id = 0):
        self.real_init = real_init
        self.steps = 0
        self.target_lane = None
        self.lane_change_active = False
        self.is_drunk = False
        self.target_speed = configs.CAR_INITIAL_VEL
        self.speed_change_active = False
        self.target_lane_center_y = None
        self.id = id
        self.just_finished_lane_change = False

        if real_init:
            self.actor = Actor(...)
            self.critic = Critic(...)
        else:
            print(f"Initialize agent Controller {self.id}")

    def make_decision(self, obs):
        self.steps += 1
        agent_lane = int(obs[0])
        agent_speed = obs[1]
        agent_heading = obs[2]
        car_lane = int(obs[3])
        car_speed = obs[4]
        car_yaw = obs[5]
        dx = obs[6]
        dy = obs[7]
        lane_error = obs[8:12]
        target_lane_error = None

        # print(f"OBS: lane={agent_lane}, speed={agent_speed:.1f}, heading={agent_heading:.2f}, car_lane={car_lane}, car_speed={car_speed:.1f}, car_yaw={car_yaw:.2f}, dx={dx:.1f}, dy={dy:.1f}, lane_error={lane_error}")

        if self.is_drunk:
            # every 20 steps repeat, heading from -30 to 30 degrees
            turn_cmd = (0.5 + (np.random.random() - 0.5)*0.3) * np.sin(2 * np.pi * self.steps / 20 * (self.steps // 20 % 2 * 2 - 1)) # alternate between left and right turns
            # every 60 steps repeat, speed change from -10 to 10
            acc_cmd = 10.0 * np.sin(2 * np.pi * self.steps / 60)
            return turn_cmd, acc_cmd
        
        if self.steps < configs.G_STEPS * 0.15 and not configs.EVAL:
            return 0.0, 0.0 # avoid hurting early learning ## don't hurt me, don't hurt me, no more

        if self.real_init:
            # Process obs and get action from actor
            action = self.actor(obs)
            turn_cmd, acc_cmd = action[0], action[1]
            return turn_cmd, acc_cmd
        else:
            turn_cmd = 0.0
            acc_cmd = 0.0
            # detect for lane change every 150 steps
            if (not self.lane_change_active) and (self.steps % 150 == 0):
                if (car_lane != agent_lane) or np.random.rand() < 0.2: # small chance to change out
                    self.lane_change_active = True
                    # print(f"Start lane change for agent {self.id}")
                    if agent_lane == 0:
                        self.target_lane = 1
                    elif agent_lane == 1:
                        self.target_lane = 0
                    elif agent_lane == 2:
                        self.target_lane = 3
                    elif agent_lane == 3:
                        self.target_lane = 2
            # move to the chosen lane
            if self.lane_change_active:
                # print(f"Changing lane from {agent_lane} to {self.target_lane}")
                target_lane_error = lane_error[self.target_lane]

                lane_tolerance = 5.0
                heading_tolerance = 0.03

                desired_heading = 0.003 * target_lane_error
                desired_heading = np.clip(desired_heading, -0.15, 0.15)

                if agent_heading < desired_heading - heading_tolerance:
                    turn_cmd = np.random.choice([0.07, 0.09, 0.11])
                elif agent_heading > desired_heading + heading_tolerance:
                    turn_cmd = -np.random.choice([0.07, 0.09, 0.11])
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
                    self.just_finished_lane_change = True

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
                            self.target_speed = configs.TARGET_SPEED + 20.0 # speed up even if ahead
                            # print(f"Irrational speed up for agent {self.id}")
                        else:
                            self.target_speed = configs.TARGET_SPEED - 20.0
                            # print(f"Blocking slow down for agent {self.id}")
                    else: # agent behind
                        if irrational_probs < 0.1:
                            self.target_speed = configs.TARGET_SPEED - 20.0 # slow down even if behind
                            # print(f"Irrational slow down for agent {self.id}")
                        else:
                            self.target_speed = configs.TARGET_SPEED + 10.0
                            # print(f"Chasing speed up for agent {self.id}")
                    self.speed_change_active = True
                    self.target_speed = np.clip(self.target_speed, configs.MIN_SPEED, configs.MAX_SPEED)
            if self.speed_change_active:
                speed_diff = self.target_speed - agent_speed

                if speed_diff > configs.MAX_ACC/1.5:
                    acc_cmd = configs.MAX_ACC
                    # print(f"Accelerating to {self.target_speed:.1f}")
                elif speed_diff < -configs.MAX_ACC/1.5:
                    acc_cmd = -configs.MAX_ACC
                    # print(f"Braking to {self.target_speed:.1f}")
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