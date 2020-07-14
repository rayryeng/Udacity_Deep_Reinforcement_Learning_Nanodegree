# Project 1: Navigation

### Introduction

For this project, the objective was to train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the must achieve an average score of +13 over 100 consecutive episodes.

### Getting Started

1. The `setup.ipynb` notebook in the main directory of this repo should be run so that the necessary installs can take place.  This includes installing a minimal working version of OpenAI Gym, the necessary dependencies from the Udacity Deep Reinforcement Learning Nanodegree and downloading the Unity environments for completing this project.  For completeness, the Linux and Mac OS versions of the Unity environments are included in this repo so you do not need to run those cells in the notebook but they are provided for completeness.

2.  The `Navigation.ipynb` and `Navigation_Pixels.ipynb` notebook files are to help introduce how to interact with the environment both using the 37-dimensional state space quantifying the perception of the world and with the state being represented as raw pixel values where we are presented with an image of the world in a 84 x 84 RGB image instead of the state vector.  These are primarily provided for self-containment but are not essential to running the training code and testing out the final learned agent.  These files were not provided for the final submission of the project as they were already included in the original repository and are for exploratory analysis only.

3. The `Train_Banana.ipynb` notebook file chronicles the training process to allow an agent to learn what is necessary to collect as many bananas as possible before the time runs out.  The agent is trained by using a Deep Q-Network (DQN) where the Q-function and thus the Q-table are learned by training a deep neural network.  This involves defining the Unity environment and getting the default "brain", importing in the class that implements the DQN Agent (will talk about this soon) and running the DQN training loop to finally save the learned weights for use in testing.

4. `model.py` contains the model definitions of learning the Q-function by means of a neural network.  There are two neural network definitions here depending on which state vector is chosen to represent the environment.  The first class `QNetwork` is used when the state vector is the 37-dimensional representation whereas the second class, `QNetworkConvolutional` is used when the state is represented as the 84 x 84 pixel RGB image.

5. `dqn_agent.py` contains the engine for training an agent through the DQN paradigm.  It is also used in the `Train_Banana.ipynb` notebook.  When we interact with the world and obtain the state vectors, actions and rewards we provide these to the engine that will update the neural network weights defined by the aforementioned model and eventually learn to obtain the largest number of yellow bananas possible within the allotted time.

6. `Train_Banana_Pixels.ipynb` is the attempt to train a DQN Agent by using the 84 x 84 RGB image as the state.  I am currently not able to train this due to problems with the `multiprocessing` module for this particular version of the environment.  I've tried training on both Google Colab and DeepNote which randomly freezes after a certain number of episodes as it is waiting for data that never arrives.  I've also tried training locally on my machine which has a modest GPU but the RAM usage for the Unity environment shoots upwards to 12 GB which slows down my machine immensely so it was not possible to train within an acceptable amount of time.  The structure to allow for training is there but this has not been completed.  However, this is not a requirement for completing this project.  This file was also not provided for the final submission of the project.

7.  `checkpoint.pth` is the saved weights for the neural network for the learned DQN Agent for collecting yellow bananas.

8.  `Test_Banana.ipynb` is a notebook that reloads the trained DQN Agent and runs the environment in test mode so that we can visually inspect the performance of the agent.
   
9.  The Banana directories contain the Unity environment that is used to interact with the DQN Agent.  These were also not included in the final project submission.

### Instructions

1. (Optional) Run the `setup.ipynb` notebook file to install the necessary dependencies.  This is not needed if those are already set up on your machine.  If you skip this step, you will need to move the Banana Unity environments into this directory, or modify the `Train_Banana.ipynb` notebook so that the proper Unity environment is pointed to.  There is more detail in that notebook.

2. Open up the `Train_Banana.ipynb` and simply run all of the cells.  These cells include setting up the Unity environment, setting up the DQN Agent for training, setting up the training loop and executing the training.

3.  After training, a `checkpoint.pth` file is created in this directory.  Take special care that there is already a checkpoint file (see point #7 in the Getting Started section) that contains the learned agent when I performed the training session with the agent.  You can either overwrite this file with your own weights, or you can save them by renaming the file prior to executing the training loop cell.

4.  Open up the `Test_Banana.ipynb` notebook and simply run all of the cells.  If things work out correctly, you will see a window pop up that interacts with the environment automatically and collects the yellow bananas.  Below is a video capture that shows one sample run of collecting the yellow bananas and avoiding the blue ones using the trained network after running step 3.  The score collected for this run was 21.

<p align="center">
  <img src="images/bananarun.gif" />
</p>