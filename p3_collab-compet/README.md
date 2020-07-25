# Project 3: Collaboration and Competition

## Introduction

For this project, we will work with the
[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
environment.

The example below shows a trained agent that was achieved after completing this
project.

![Trained Agent](images/pingpong.gif)

In this environment, two agents control rackets to bounce a ball over a net. If
an agent hits the ball over the net, it receives a reward of +0.1.  If an agent
lets a ball hit the ground or hits the ball out of bounds, it receives a reward
of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and
velocity of the ball and racket. Each agent receives its own, local observation.
Two continuous actions are available, corresponding to movement toward (or away
from) the net, and jumping.  They are both represented in the range of `[-1,1]`.

The task is episodic, and in order to solve the environment, your agents must
get an average score of +0.5 (over 100 consecutive episodes, after taking the
maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without
  discounting), to get a score for each agent. This yields 2 (potentially
  different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of
those **scores** is at least +0.5.

## Environment

For this project, we were already provided a compiled environment and all we
have to do is download it for use.  Please go to the `setup.ipynb` notebook in
the main directory of this repo to run the cells to download the files necessary
for this project.  Interestingly, because there are two agents and thus
qualifying as a multi-agent system, we can actually use the entire training and
testing ecosystem from the previous project (Project #2 - Reacher) using the
DDPG algorithm to train our agents.  The only thing that will need carefully
consideration and selection is the hyperparameters.

## Getting Started

1. The `setup.ipynb` notebook in the main directory of this repo should be run
   so that the necessary installs can take place.  This includes installing a
   minimal working version of OpenAI Gym, the necessary dependencies from the
   Udacity Deep Reinforcement Learning Nanodegree and downloading the Unity
   environments for completing this project.  For completeness, the Linux and
   Mac OS versions of the Unity environments are included in this repo so you do
   not need to run those cells in the notebook but they are provided just in
   case.  This file was not provided for the final submission of the project.

2.  The `Tennis.ipynb` notebook file is to help
    introduce how to interact with the Unity environment where
    the states are quantified using `N` dimensional vectors.  In the case of
    `Tennis.ipynb`, the Tennis environment's state space for each agent is
    represented by an 8-dimensional vector which quantifies the state of a
    tennis racket in the environment.  Take note that there are three stacked
    observations at each iteration, meaning that there are three consecutive
    observations at each iteration thus making the state space a 24-dimensional
    vector.  This is primarily provided for
    self-containment but are not essential to running the training code and
    testing out the final learned Tennis agent.  This file was not provided
    for the final submission of the project as it was e already included in the
    original repository and is for exploratory analysis only.

3. The `Train_Tennis.ipynb` notebook file chronicles the training process to
   allow the two Tennis agents to learn what is necessary solve the Tennis
   environment.  The agent is trained by using the DDPG algorithm where the
   Actor and Critic mechanisms were learned
   by training a deep neural network.  This involves defining the Unity
   environment and getting the default "brain", importing in the class that
   implements the DDPG Agent (will talk about this soon) and running the DDPG
   training loop to finally save the learned weights for use in testing.

4. `model.py` contains the model definitions for the Actor and Critic, both
   represented as a neural network in the `Actor` and `Critic` classes
   respectively.

5. `ddpg_agent.py` contains the engine for training an Actor-Critic system
   through the DDPG algorithm.  It is also used in the `Train_Tennis.ipynb`
   notebook.  When we interact with the world and obtain the state vectors,
   actions and rewards we provide these to the engine so that it will update the
   neural network weights defined by the aforementioned models and eventually
   learn to solve the Reacher environment.

6.  `checkpoints_tennis` is a directory containing the saved weights for the DDPG Actor
    and Critic, which are in `checkpoints_tennis/checkpoint_actor.pth` and
    `checkpoints_tennis/checkpoint_critic.pth` respectively.

7.  `Test_Tennis.ipynb` is the are notebook that
    reloads the trained DDPG Actor-Critic and runs the environment in test mode
    so that we can visually inspect the performance of the two agents.

8.  The Tennis directories contain the Unity environment that is used to
    interact with the DDPG Actor-Critic.  These were also not included in the
    final project submission.

## Instructions

1. (Optional) Run the `setup.ipynb` notebook file to install the necessary
   dependencies.  This is not needed if these are already set up on your
   machine.  If you skip this step, you will need to move the Reacher Unity
   environments into this directory, or modify the `Train_Tennis.ipynb`
   notebook so that the proper Unity environment is pointed to.  There is more
   detail in that notebook.

2. Open up the `Train_Tennis.ipynb` and simply run all of the cells.  These
   cells include setting up the Unity environment, setting up the DQN Agent for
   training, setting up the training loop and executing the training.

3.  After training, a `checkpoints_tennis` directory is created which will store the
    weights for the Actor and Critic.  Take special care that there are already
    checkpoints (see point #6 in the Getting Started section) that contains the
    learned DDPG Actor and Critic when I performed a training session.  You can
    either overwrite this file with your own weights, or you can save them by
    renaming the file prior to executing the training loop cell.

4.  Open up the `Test_Tennis.ipynb` notebook and simply run all of the cells.
    This notebook operates in headless mode in a Linux environment.  There is a
    `Test_Tennis_MacOS.ipynb` notebook file that operates the same way but for
    Mac OS.  If things work out correctly, you will see a window pop up that
    with the environment and with the two agents.  The video capture above shows
    one sample run of solving the environment after running steps #3 and #4 in
    this section.  The final score above was ~+2.6 which is roughly the maximum
    possible score one can achieve if the ball never hits the ground and thus
    reaching the maximum number of timestamps for this environment which is 1000.
