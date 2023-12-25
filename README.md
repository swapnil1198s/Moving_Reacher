#Moving Reacher Environment

https://github.com/swapnil1198s/Moving_Reacher/assets/46658528/178950f3-b837-4834-b438-6e40a79b5491

##Overview
The Moving Reacher is a custom reinforcement learning environment built using Pygame and Gymnasium (formerly known as Gym). It simulates a 2D robotic arm (reacher) with two segments, each controlled by applying torques at hinge joints. The environment is designed for experiments with reinforcement learning algorithms, particularly those focusing on continuous control tasks.

##Features
2D Robotic Arm Simulation: Simulates a two-segment robotic arm with joints controlled by torque.
Customizable Parameters: Arm length, torque limits, and other parameters can be easily adjusted.
Target Following: The arm aims to reach or follow a moving target, which follows a sinusoidal path.
Observation Space: Includes arm position, joint angles, and the target's position.
Action Space: Continuous action space representing the torques applied to the arm's joints.
Reward Function: Customizable reward function based on the distance between the arm's tip and the target.

##Dependencies
Python 3.x
Pygame
Gymnasium (formerly Gym)
NumPy
