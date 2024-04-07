import gym
import random

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        pass

    def act(self, observation):
        right_action = [0,1,2,3,4,5,10,11]
        action = random.choice(right_action)
        return action    
        # return self.action_space.sample()
