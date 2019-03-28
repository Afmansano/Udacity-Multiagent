# Project 3: Collaboration and Competition

### Introduction

This is the impelementation of Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Collaboration and Competition project. 

For the environment we use the [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) Tennis environment.

In the Tennis environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Each agent shall keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

![tennis_env](images/tennis.png)

The image above depicts the environment.

There are two agents, each one observe one state. 

#### State Space

The environment state space for each agent is composed of a 24-dimensional np-array.:

```python
States look like: [ 0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -7.389936 , -1.5, -0,  0,  6.8317208 ,  5.9960761 , -0,  0]
```

corresponding to position and velocity of ball and racket.


#### Action Space

For this environment we have 2 different continuous actions, corresponding to movement toward net or away from net, and jumping.


#### Solving the environment

The task is episodic, and the environment is considered solved the when agents get an average score of +0.5 over 100 consecutive episodes (after taking the maximum over both agents).

## Getting started

### Installation requirements

This project was developed using Udacity workspace. If you are using this environment just need to run the command:

!pip -q install ./python

Otherwise, if you are running locally, follow the instructions bellow:

- You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)

- Then you have to install the Unity environment The environment can be downloaded from one of the links below for all operating systems:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


### Train the agent
    
Execute the Tennis.ipynb notebook within this Nanodegree Udacity Online Workspace for "project #2  Continuous Control" (or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code )
