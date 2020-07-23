# Project 2: Continuous Control

## Introduction

For this project, we will work with the
[Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
environment.

The example below shows a trained agent that was achieved after completing this
project.

![Trained Agent](images/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward
of +0.1 is provided for each step that the agent's hand is in the goal location.
Thus, the goal of an agent in the environment is to maintain its position at the
target location for as many time steps as possible.  The observation space
consists of 33 variables corresponding to position, rotation, velocity, and
angular velocities of the arm. Each action is a vector with four numbers,
corresponding to torque applicable to two joints. Every entry in the action
vector should be a number between -1 and 1.

## Environment

For this project, wer were provided with two separate versions of the Unity
environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the
  environment. 

The second version is useful for algorithms like
[PPO](https://arxiv.org/pdf/1707.06347.pdf),
[A3C](https://arxiv.org/pdf/1602.01783.pdf), and
[D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple
(non-interacting, parallel) copies of the same agent to distribute the task of
gathering experience.

In this project, we will use the [Deep Deterministic Policy Gradient (DDPG)
algorithm](https://arxiv.org/abs/1509.02971) method to solve this environment.
In addition, the second version of the environment was chosen for this project.
DDPG originally was for a single agent in an Actor-Critic scenario but we will
adapt this to use information from the 20 agents to solve the environment.

## Solving the Environment

In order to deem the environment as successful in solving it, the agents must
collectively achieve an average score of +30 over 100 consecutive episodes.
Specifically:
- After each episode, we add up the rewards that each agent received (without
  discounting) to get a score for each agent.  This yields 20 (potentially
  different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode where the average is over
  all 20 agents.
- We thus examine this average score over 100 consecutive episodes and if the
  average of these average scorse over +30, we will deem the environment as
  successfully being solved.

## Note regarding the Crawler Environment

As an additional challenge, the task of solving the Crawler environment was
available which could use the same implementation to solve the Reacher
environment.  This was an optional task and I have opted not to solve it here -
not without a lack of trying of course.  The navigation and training notebooks
(to be discussed later) are made available but a full model was not trained to
solve this environment.  After ~26800 iterations, the average reward blew up to
NaN and haven't bothered to try training it again.  We will not discuss this
environment in the final report beyond what is being discussed here.

## Getting Started

1. The `setup.ipynb` notebook in the main directory of this repo should be run
   so that the necessary installs can take place.  This includes installing a
   minimal working version of OpenAI Gym, the necessary dependencies from the
   Udacity Deep Reinforcement Learning Nanodegree and downloading the Unity
   environments for completing this project.  For completeness, the Linux and
   Mac OS versions of the Unity environments are included in this repo so you do
   not need to run those cells in the notebook but they are provided just in
   case.  This file was not provided for the final submission of the project.

2.  The `Continuous_Control.ipynb` and `Crawler.ipynb` notebook files is to help
    introduce how to interact with their corresponding Unity environments where
    the states are quantified using `N` dimensional vectors.  In the case of
    `Continuous_Control.ipynb`, the Reacher environment's state space is
    represented by a 33-dimensional vector which quantifies the state of the
    agent's arm during an episode.  These are primarily provided for
    self-containment but are not essential to running the training code and
    testing out the final learned Reacher agent.  These files were not provided
    for the final submission of the project as they were already included in the
    original repository and are for exploratory analysis only.

1. The `Train_Reacher.ipynb` notebook file chronicles the training process to
   allow the 20 Reacher agents to learn what is necessary solve the Reacher
   environment.  The agent is trained by using a the Deep Deterministic Policy
   Gradient (DDPG) algorithm where the Actor and Critic mechanisms were learned
   by training a deep neural network.  This involves defining the Unity
   environment and getting the default "brain", importing in the class that
   implements the DDPG Agent (will talk about this soon) and running the DDPG
   training loop to finally save the learned weights for use in testing.

2. `model.py` contains the model definitions for the Actor and Critic, both
   represented as a neural network in the `Actor` and `Critic` classes
   respectively.

3. `ddpg_agent.py` contains the engine for training an Actor-Critic system
   through the DDPG algorithm.  It is also used in the `Train_Reacher.ipynb`
   notebook.  When we interact with the world and obtain the state vectors,
   actions and rewards we provide these to the engine so that it will update the
   neural network weights defined by the aforementioned models and eventually
   learn to solve the Reacher environment.

4.  `checkpoints` is a directory containing the saved weights for the DDPG Actor
    and Critic, which are in `checkpoints/checkpoint_actor.pth` and
    `checkpoints/checkpoint_critic.pth` respectively.

5.  `Test_Reacher.ipynb` and `Test_Reacher_MacOS.ipynb` are notebooks that
    reload the trained DDPG Actor-Critic and runs the environment in test mode
    so that we can visually inspect the performance of the 20 agents.

6.  The Reacher directories contain the Unity environment that is used to
    interact with the DDPG Actor-Critic.  These were also not included in the
    final project submission.

## Instructions

1. (Optional) Run the `setup.ipynb` notebook file to install the necessary
   dependencies.  This is not needed if these are already set up on your
   machine.  If you skip this step, you will need to move the Reacher Unity
   environments into this directory, or modify the `Train_Reacher.ipynb`
   notebook so that the proper Unity environment is pointed to.  There is more
   detail in that notebook.

2. Open up the `Train_Reacher.ipynb` and simply run all of the cells.  These
   cells include setting up the Unity environment, setting up the DQN Agent for
   training, setting up the training loop and executing the training.

3.  After training, a `checkpoints` directory is created which will store the
    weights for the Actor and Critic.  Take special care that there are already
    checkpoints (see point #4 in the Getting Started section) that contains the
    learned DDPG Actor and Critic when I performed a training session.  You can
    either overwrite this file with your own weights, or you can save them by
    renaming the file prior to executing the training loop cell.

4.  Open up the `Test_Reacher.ipynb` notebook and simply run all of the cells.
    This notebook operates in headless mode in a Linux environment.  There is a
    `Test_Reacher_MacOS.ipynb` notebook file that operates the same way but for
    Mac OS.  If things work out correctly, you will see a window pop up that
    with the environment and with the 20 agents.  The video capture above shows
    one sample run of solving the environment after running steps #3 and #4 in
    this section.  The final score in the above video capture was ~30.
