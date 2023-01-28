import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits import mplot3d
import time 
import math
from tqdm import tqdm
import pandas as pd
'''
This cell will load the environment and select the render mode for the simulation, since it is not time consuming; 
the human node rendering is intended for the real-time simulation of the system (reported at the end of the monte-carlo section)
'''

class CartPole_v1():
    
    def __init__(self, n_obs = 1000, n_split = [4,5,10,12], sim = True, PLOT_DEBUG = False, with_experience = False):
        self.env_name = 'CartPole-v1'
        if(sim):
            self.env =  gym.make(self.env_name, render_mode='rgb_array')   # for simulation
        else: 
            self.env = gym.make(self.env_name, render_mode='human')         # for rendering    
        self.n_split = n_split 
        self.n_obs = n_obs
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]
        self.intervals = []
        self.PLOT_DEBUG = PLOT_DEBUG
        self.experience = with_experience
        
        self.discrete_bucket()

    def making_experience(self):
        print("making some experience running {} episodes ... ".format(self.n_obs))
        
        '''
        As it is reported on the official documentation for this environment, the observation space is composed
        of four continous time variable. In order to apply RL method such as MC it is nedeed to espress this state in a discrete world.
        This cell analyze which are the most common output coming from the environment running 000 action in different states. 
        '''
        obs_list = []
        observation, info = self.env.reset()
        obs_list.append(observation)

        #here we are observaing a sequence of 10000 actions took, without considering the number of episode
        for _ in range(1000):
            action = self.env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = self.env.step(action)
            obs_list.append(observation)
            if terminated or truncated:
                observation, info = self.env.reset()
        print("done") 
        return obs_list
    
    def intervals_split(self, start, finish, parts):
            '''
            function that, given an interval and the number of split to apply, 
            return a list of intervals equalli separated by the number of split given as input
            '''
            part_duration = (finish-start) / parts
            return [start+i * part_duration for i in range(parts+1)]
        
    def discrete_bucket(self):
        
        if(self.experience):
            obs_list = self.making_experience()
            print("discretizing the environment ... ")
            states = [[], [], [], []]
            for k in range(len(obs_list)):
                states[0].append(obs_list[k][0])
                states[1].append(obs_list[k][1])
                states[2].append(obs_list[k][2])
                states[3].append(obs_list[k][3])
                
            # took the extrema from the simulations 
            extrema = []
            '''
            This is an important parameter to set: increasing the number of split we are more accurate into discretize the 
            continous state space associate to the observations. However, this will negatively have an impact on the number 
            of episode needed by the algorithm to learn a correct policy
            '''
            for k in range(len(states)):
                extrema.append([np.min(states[k]), np.max(states[k])])
                self.intervals.append(self.intervals_split(extrema[k][0], extrema[k][1], self.n_split[k]))
            if(self.PLOT_DEBUG):   
                y = [0,0.5,1,1.5]
                for k in range(len(obs_list)):
                    plt.plot(obs_list[k][0], y[0], 'o', color='gray', alpha = 0.05)
                    plt.plot(obs_list[k][1], y[1], 'o', color='gray', alpha = 0.05)
                    plt.plot(obs_list[k][2], y[2], 'o', color='gray', alpha = 0.05)
                    plt.plot(obs_list[k][3], y[3], 'o', color='gray', alpha = 0.05)
                    
                for k in range(len(states)):
                    for i in range(len(self.intervals[k])):
                        plt.plot(self.intervals[k][i], y[k], '|', color = 'red', markersize=5)
                        
            
        else:
            print("discretizing the environment ... ")
            # approximated value of the state of the environment
            
            self.intervals.append(np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.n_split[0]+1)[1:-1]) # state 0
            self.intervals.append(np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.n_split[1]+1)[1:-1])     # state 1
            self.intervals.append(np.linspace(self.lower_bounds[2], self.upper_bounds[2], self.n_split[2]+1)[1:-1]) # state 2
            self.intervals.append(np.linspace(self.lower_bounds[3], self.upper_bounds[3], self.n_split[3]+1)[1:-1])     # state 3
            if(self.PLOT_DEBUG):   
                y = [0,0.5,1,1.5]
                for k in range(4):  # n° of states
                    for i in range(len(self.intervals[k])):
                        plt.plot(self.intervals[k][i], y[k], '|', color = 'red', markersize=20)
                plt.axhline(y[0], color='gray', label='env threshold = 195.0', linestyle='--')
                plt.axhline(y[1], color='gray', label='env threshold = 195.0', linestyle='--')
                plt.axhline(y[2], color='gray', label='env threshold = 195.0', linestyle='--')
                plt.axhline(y[3], color='gray', label='env threshold = 195.0', linestyle='--')       

            plt.show()
            
        print("done")

        
    def state_projection(self, value):
        '''
        This function associate to each state observation a unique positive integer value, useful
        for constructing the Q_table
        '''
        discrete_state = []
        for k in range(len(value)):
            discrete_state.append(np.digitize(value[k], self.intervals[k])) # the k-th state description with the k-th intervals split
        discrete_state = discrete_state[0]+discrete_state[1]*(self.n_split[0])+discrete_state[2]*(self.n_split[0])*(self.n_split[1])+discrete_state[3]*(self.n_split[0])*(self.n_split[1])*(self.n_split[2])
        
        return discrete_state