
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition


This project will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

This project is based on [DDPG Bipedal](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal)

###  Author Aysun Akarsu

###  Objective

This project's aim is to train two reinforcement learning agents to control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Document's objective 

This document's objective is to explain which algorithms, methods, model architetures and hyper parameters are used to realize the project. 


## Learning algorithm

### Method
There are 3 methods which can be used in reinforcement algorithms:<br>
Values-Based Methods: Aim to approximate the optimal value function, which is a mapping between an action and a value. With these methods the best action (the action with the greatest value) for each state is found. 
Policy-Based Methods: Optimize the strategy directly without using a value function. It is good for the continuous or stochastic action space. They have faster convergence.
Hybrid Methods: Combination of value-based and policy methods which overcome the drawbacks of both of them.

In this project a "hybrid method", the actor critic method is used. <br>
This method uses two neural networks; actor and critic:<br>
Critic measures the quality of the action taken based on value<br>
Actor controls how the agent behaves based on policy<br>

### Algorithm

This project uses DDPG algorithm (deep deterministic policy gradient)with actor critic method that uses two agents playing tennis in order to win against each other. The DDPG algorithm uses two additional mechanisms: the replay buffer and the progressive updates.


### Hyper parameters

BUFFER_SIZE = int(1e5) # replay buffer size<br>
BATCH_SIZE = 256       # minibatch size <br>
GAMMA = 0.99           # discount factor <br>
TAU = 2e-3             # for soft update of target parameters <br>
LR_ACTOR = 2e-4        # learning rate of the actor <br>
LR_CRITIC = 3e-3       # learning rate of the critic <br>
WEIGHT_DECAY = 0       # L2 weight decay <br>
OU_SIGMA  = 0.01  #Ornstein-Uhlenbeck noise parameters sigma <br>
OU_THETA  = 0.15  #Ornstein-Uhlenbeck noise parameters theta<br>
     

### Model architecture

The algorithm uses two deep neural networks (actor-critic) with the following architecture:

#### Actor
Actor(<br>
  (bn1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) <br>
  (fc1): Linear(in_features=24, out_features=216, bias=True) <br>
  (bn2): BatchNorm1d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (fc2): Linear(in_features=216, out_features=128, bias=True)<br>
  (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (fc3): Linear(in_features=128, out_features=2, bias=True)<br>
)<br>

#### Critic
Critic(<br>
  (bn1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (fcs1): Linear(in_features=24, out_features=216, bias=True)<br>
  (bn2): BatchNorm1d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (fc2): Linear(in_features=218, out_features=128, bias=True)<br>
  (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (fc3): Linear(in_features=128, out_features=1, bias=True)<br>
)<br>

#### Training

In the training phase of this learning algorithm, actor critic methods are used.In these methods two neural networks are used to learn a policy (Actor) and a value function (Critic). 

Episode 100	Average Score: 0.019 <br>
Episode 200	Average Score: 0.033 <br>
Episode 300	Average Score: 0.108 <br>
Episode 362	Average Score: 0.511 <br>
Environment solved in 262 episodes!	Average Score: 0.51

### Plot of rewards

## Average score per episode

![plot_of_rewards](https://raw.githubusercontent.com/aysunakarsu/udacity_drlnd_tennis/master/plot_of_rewards_01.png)<br>

## Average score per 100 episodes

![plot_of_rewards_100](https://raw.githubusercontent.com/aysunakarsu/udacity_drlnd_tennis/master/plot_of_rewards_100_01.png)<br>


### Improvements

Hyperparameters can be tuned<br>
Some more layers to the neural networks Actor and Critic can be added. <br>
Different hidden layer sizes can be used.<br>
Other algorithms  can be considered, such as PPO.<br>
