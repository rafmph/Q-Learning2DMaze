# -*- coding: utf-8 -*-
import pygame
pygame.display.set_mode((640,480))
import sys
import numpy as np
import math
import random
import gym
import gym_maze
import sys

#Method for finding written arguments in terminal
def get_argument(arg_keyword, default_value):
    sys_args = sys.argv[1:]
    if arg_keyword in sys_args:
        argument_value = sys_args[sys_args.index(arg_keyword)+1]
    else:
        argument_value = default_value
    return argument_value

# Class for executing path optimization within our environment
class QLearningMaze():

  def select_action(self, state, explore_rate):
      # Select a random action
      if random.random() < explore_rate:
          action = self.env.action_space.sample()
      # Select the action with the highest q
      else:
          action = int(np.argmax(self.q_table[state]))
      return action

      #Choose between the least minimum explore rate chosen or the one dictated by the model using discount rate
      #This rate will determine how much the algorithm looks for new paths
  def get_explore_rate(self, t, min_explore_rate, discount_rate):
      return max(min_explore_rate, min(0.8, 1.0 - math.log10((t+1)/discount_rate)))

      #Choose between the least learning rate chosen or the one dictated by the model using discount rate
      #This rate will determine how much the algorithm looks for old paths
  def get_learning_rate(self, t, min_learning_rate, discount_rate):
      return max(min_learning_rate, min(0.8, 1.0 - math.log10((t+1)/discount_rate)))

      #Method for converting observation into specific index within Height x Width in maze
  def state_to_bucket(self, state, maze_shape, maze_bounds):
      bucket_indice = []
      for i in range(len(state)):
          if state[i] <= maze_bounds[i][0]:
              bucket_index = 0
          elif state[i] >= maze_bounds[i][1]:
              bucket_index = maze_shape[i] - 1
          else:
              bound_width = maze_bounds[i][1] - maze_bounds[i][0]
              offset = (maze_shape[i]-1)*maze_bounds[i][0]/bound_width
              scaling = (maze_shape[i]-1)/bound_width
              bucket_index = int(round(scaling*state[i] - offset))
          bucket_indice.append(bucket_index)
      return tuple(bucket_indice)

  def __init__(self, env, num_episodes, total_iterations, min_explore_rate, min_learning_rate, discount_rate, render_maze = True, accuracy_goal=None):
    self.env = gym.make(env)
    #Up, down, left and right are 4 valid actions in environment
    self.num_actions = self.env.action_space.n
    #Episodes equals number of epochs or times agent reaches goal
    self.num_episodes = num_episodes
    #Height x Width grid
    self.maze_shape = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))
    #Number of iterations run before agent optimizes next epoch
    self.total_iterations = np.prod(self.maze_shape, dtype=int) * total_iterations
    self.iterations_end =  total_iterations
    self.iterated_so_far = np.prod(self.maze_shape, dtype=int)
    #Used for indexing and giving low reward if out of bounds
    self.maze_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
    #Discount return on every action to minimize amount of actions needed
    self.discount_rate = np.prod(self.maze_shape, dtype=float) / discount_rate
    #Creating a Q-Table for each state-action pair
    self.q_table = np.zeros((self.maze_shape[0], self.maze_shape[1], self.num_actions), dtype=float)
    #Hyperparameter for exploring new actions
    self.min_explore_rate = min_explore_rate
    #Hyperparameter for choosing defined actions instead of exploring
    self.min_learning_rate = min_learning_rate
    #True of False variable for rendering maze
    self.render_maze = render_maze
    #Finish optimizing once specific reward is met
    self.accuracy_goal = accuracy_goal


  def simulate(self):

      # Instantiating the learning related parameters
      explore_rate = self.get_explore_rate(0, self.min_explore_rate, self.discount_rate)
      learning_rate = self.get_learning_rate(0, self.min_learning_rate, self.discount_rate)
      discount_factor = 0.99

      num_streaks = 0

      # Render tha maze
      self.env.render()

      for episode in range(self.num_episodes):

          # Reset the environment
          obv = self.env.reset()

          # the initial state
          state_0 = self.state_to_bucket(obv, self.maze_shape, self.maze_bounds)
          total_reward = 0

          for t in range(self.total_iterations):

              # Select an action
              action = self.select_action(state_0, explore_rate)

              # execute the action
              obv, reward, done, _ = self.env.step(action)

              # Observe the result
              state = self.state_to_bucket(obv, self.maze_shape, self.maze_bounds)
              total_reward += reward

              # Update the Q based on the result
              best_q = np.amax(self.q_table[state])
              self.q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - self.q_table[state_0 + (action,)])

              # Setting up for the next iteration
              state_0 = state

              # Render tha maze
              if self.render_maze:
                  self.env.render()

              if self.env.is_game_over():
                  sys.exit()

              if done:
                  print("Episode %d finished after %f time steps with total reward = %f (streak %d). " % (episode, t, total_reward, num_streaks))

                  if t <= self.iterated_so_far:
                      num_streaks += 1
                  else:
                      num_streaks = 0
                  break

              elif t >= self.total_iterations - 1:
                  print("Episode %d timed out at %d with total reward = %f." % (episode, t, total_reward))

          # It's considered done when it's solved over 100 times consecutively
          if num_streaks > self.iterations_end:
              break

          # Update parameters
          explore_rate = self.get_explore_rate(episode, self.min_explore_rate, self.discount_rate)
          learning_rate = self.get_learning_rate(episode, self.min_learning_rate, self.discount_rate)

          #Stop iterations if agent rewards reaches a specific score
          if self.accuracy_goal != None and total_reward >= self.accuracy_goal:
            break

if __name__ == "__main__":

    #Method for handling sys arg arguments
    num_episodes = int(get_argument("--episodes", 50000))
    total_iterations = int(get_argument("--iterations", 200))
    min_explore_rate = float(get_argument("--explore_r", 0.001))
    min_learning_rate = float(get_argument("--learn_r", 0.2))
    discount_rate = int(get_argument("--discount_r", 10))

    #Welcome message with options to choose maze type
    text_introduction = "Hey! This is a Reinforcement Learning technique for finding the optimal route for solving a maze.\nChoose between one of the following integers: 1-12\n1: 2x2 Same structure maze without portals\n2: 3x3 Same structure maze without portals\n3: 3x3 Random structure maze without portals\n4: 5x5 Same structure maze without portals\n5: 5x5 Random structure maze without portals\n6: 10x10 Same structure maze without portals\n7: 10x10 Random structure maze without portals\n8: 10x10 Random structure maze WITH portals\n9: 20x20 Random structure maze WITH portals\n10: 30x30 Random structure maze WITH portals\n11: 100x100 Same structure maze without portals\n12: 100x100 Random structure maze without portals\n"
    index_enviroment = int(input(text_introduction))
    maze_enviroments = ["maze-v0", "maze-sample-3x3-v0","maze-random-3x3-v0","maze-sample-5x5-v0","maze-random-5x5-v0",
    "maze-sample-10x10-v0","maze-random-10x10-v0","maze-random-10x10-plus-v0","maze-random-20x20-plus-v0","maze-random-30x30-plus-v0",
    "maze-sample-100x100-v0","maze-random-100x100-v0"]

    #Error handling in case users an invalid environment
    if index_enviroment >= 1 and index_enviroment <= 12:
        chosen_enviroment = maze_enviroments[index_enviroment-1]
    else:
        raise ValueError("Invalid option. Choose a number between 1 and 12 following time")

    #Start our instance for creating our maze and finding optimal path. Enter our hyperparameters
    q_learning_instance = QLearningMaze(chosen_enviroment, num_episodes, total_iterations, min_explore_rate, min_learning_rate, discount_rate, render_maze=True, accuracy_goal=None)
    #Start
    q_learning_instance.simulate()
