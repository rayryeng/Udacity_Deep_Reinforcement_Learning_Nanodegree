# Udacity Deep Reinforcement Learning Engineer Nanodegree Course


This repository contains my project work for Udacity's Deep Reinforcement
Learning Nanodegree Program.  To run the code in this repo, there is some
package setup required.  Please go to the `setup.ipynb` notebook file and run
the cells to get started.  The notebook relies on the `python` directory in this
repo so please ensure this is present before running the notebook.

## Project #1 - Navigation using Deep Q-Networks (DQNs)

For this project, the objective was to train an agent to navigate (and collect
bananas!) in a large, square world.  A reward of +1 is provided for collecting a
yellow banana, and a reward of -1 is provided for collecting a blue banana.
Thus, the goal of the agent is to collect as many yellow bananas as possible
while avoiding blue bananas.  The approach was to use a deep neural network to
help learn the optimal policy and thus the necessary actions given the
environment to obtain as many yellow bananas and avoiding as many blue bananas
as possible.  Please navigate to the `p1_navigation` directory where a
`README.md` file is present to guide you through how to set up the environment
and train your own DQN to do this task.  As a reference, a pre-trained DQN model
is present if you want to start using the framework immediately.  The learning
agent, model and the necessary Unity environment files to get the training and
testing going are also present in the aforementioned directory.  In addition,
the final report for the project can be found in this directory and is available
in markdown form and PDF form as `Report.md` and `Report.pdf` respectively.

## Project #2 - Solve the Reacher Environment using Deep Deterministic Policy Gradients (DDPG)

In this project, the objective was to train an Actor-Critic system to
automatically solve the Reacher environment provided by Unity.  In this
environment, a double-jointed arm can move to target locations. A reward of +0.1
is provided for each step that the agent's hand is in the goal location.  Thus,
the goal of an agent in the environment is to maintain its position at the
target location for as many time steps as possible.  The observation space
consists of 33 variables corresponding to position, rotation, velocity, and
angular velocities of the arm. Each action is a vector with four numbers,
corresponding to torque applicable to two joints.  Please navigate to the
`p2_continuous-control` directory where a `README.md` file is present to guide
you through how to set up the environment and train your own DDPG framework to do this
task.  As a reference, the pre-trained DDPG Actor and Critic networks
are present if you want to start using the framework immediately.  The
Actor-Critic models the necessary Unity environment files to get the
training and testing going are also present in the aforementioned directory.  In
addition, the final report for the project can be found in this directory and is
available in markdown form and PDF form as `Report.md` and `Report.pdf`
respectively.

## Project #2 - Solve the Tennis Environment using Deep Deterministic Policy Gradients (DDPG)

In this project, the objective was to train an Actor-Critic system to
automatically solve the Tennis environment provided by Unity.  In this
environment, two agents control rackets to bounce a ball over a net. If an agent
hits the ball over the net, it receives a reward of +0.1.  If an agent lets a
ball hit the ground or hits the ball out of bounds, it receives a reward of
-0.01.  Thus, the goal of each agent is to keep the ball in play.  The
observation space consists of 8 variables corresponding to the position and
velocity of the ball and racket. Each agent receives its own, local observation.
Two continuous actions are available, corresponding to movement toward (or away
from) the net, and jumping.  They are both represented in the range of `[-1,1]`.
The task is episodic, and in order to solve the environment, the agents must
get an average score of +0.5 (over 100 consecutive episodes, after taking the
maximum over both agents). Please navigate to the `p3_collab-compet`
directory where a `README.md` file is present to guide you through how to set up
the environment and train your own DDPG framework to do this task.  As a
reference, the pre-trained DDPG Actor and Critic networks are present if you
want to start using the framework immediately.  The Actor-Critic models the
necessary Unity environment files to get the training and testing going are also
present in the aforementioned directory.  In addition, the final report for the
project can be found in this directory and is available in markdown form and PDF
form as `Report.md` and `Report.pdf` respectively.