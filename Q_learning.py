# Implement Q-Learning algorithm for reinforcement learning
# use the environment FrozenLake-v1 from Farama Gymnasium
# explore tuning parameters such as learning rate, discount factorm and epsilon decay rate

import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .4
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    #  update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]


        while (not done):
            # Epsilon-greedy policy for action selection
            #if Epsilon is still > 1, the robot will explore (take a random action)
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  
            else:
                # Exploit: choose the best known action (greedy)
                prediction = np.array([Q_table[(obs, a)] for a in range(env.action_space.n)])
                action = np.argmax(prediction)
                
             # Take the action and observe the outcome
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Q-learning update
            if not done:
                # Get the max future Q value
                future_max_reward = np.max([Q_table[(new_obs, a)] for a in range(env.action_space.n)])
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + \
                                          LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_max_reward)
            else:
                # If the episode is done, there is no future, so we only consider the immediate reward
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + \
                                          LEARNING_RATE * reward
                                          
            episode_reward += reward  # update episode reward
            obs = new_obs  # move to the new state
            
            # # Decay epsilon
            # EPSILON = max(EPSILON * EPSILON_DECAY, 0.01)  # Ensure epsilon is never less than 0.01
        EPSILON = EPSILON * EPSILON_DECAY

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
  