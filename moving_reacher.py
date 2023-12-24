import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gym.envs.mujoco import MujocoEnv
from gym import utils

class MovingReacher(MujocoEnv, utils.EzPickle, gym.Env):
    # Parameters for object
    metadata = {
        "render_modes":[
            "human", 
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 20,
    }

    def __init__(self, render_mode, size = 50):
        self.window_size = 800 
        
        # We are concerned about the arm pose, target position, and the distance between the tip and target positions
        """ 
        ###Arm
            arm [0] = position_x
            arm [1] = position_y
            arm [2] = cos(joint0)
            arm [3] = cos(joint1) 
            arm [4] = sin(joint0) 
            arm [5] = sin(joint1)
            arm [6] = angular velocity of arm 1 
            arm [7] = angular velocity of arm 2

        ###Target
            target [0] = target_position_x
            target [1] = target_position_y
        
        ###Distance
            distance[0] = fingertip_position_x - target_position_x
            distance[1] = fingertip_position_y - target_position_y
            distance[2] = fingertip_position_z - target_position_z = 0 ---because for now we are working with a 2d world
        """
        self.observation_space = spaces.Dict(
            {
                "arm":spaces.Box(low=-np.inf, high=np.inf, shape=(8,),dtype=np.float64),
                "target":spaces.Box(low=-np.inf, high=np.inf, shape=(2,),dtype=np.float64),
                "distance": spaces.Box(low=-np.inf, high=np.inf, shape=(3,),dtype=np.float64),
            }
        )
        
        #An action `(a, b)` represents the torques applied at the hinge joints.
        self.action_space = spaces.Box(low = -1, high=1, shape=(2,), dtype=np.float32)

        assert render_mode==None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.size = size #Length of both arms of the reacher. One cylinder from base(joint0) to joint1 and the second cylinder from joint1 to fingertip
        self.arm_length1 = size  # Length of the first arm segment
        self.arm_length2 = size  # Length of the second arm segment
        self.base_position = (0,self.window_size/2) #Starts from the left side of the screen
        self.fingertip_position = (0,0)
        self.joint0 = np.pi/2 #Joint0 angle
        self.joint1 = np.pi/2 #Joint1 angle
        self.vel0 = 0 # Angular velocity of arm 1
        self.vel1 = 0 # Angular velocity of arm 2

        self.target_position = (0, self.window_size/2 - 50) #Start from the left of the screen moving horizontally along a sinusoidal path 

        self.base_vel = 5 #Speed at which the base is moving horizontally

        self.target_x_speed = 5 #speed of target along the x axis

        # Parameters for the sinusoidal movement of the target
        self.target_amplitude = 5  # Amplitude of the sinusoidal curve
        self.target_frequency = 10    # Frequency of the sinusoidal curve
        self.time_step = 0           # Time step counter


    # Convert the state to observation
    def __get_obs(self):
        return {
            "arm": [self.base_position[0], self.base_position[1], np.cos(self.joint0), np.cos(self.joint1), np.sin(self.joint0), np.sin(self.joint1), self.vel0, self.vel1],
            "target": [self.target_position[0], self.target_position[1]],
            "distance": [self.fingertip_position[0]-self.target_position[0], self.fingertip_position[1]-self.target_position[1], 0] #Z axis distance is 0 for now as we are working with a 2D world right now
        }

    # Convert the state to info
    def __get_info(self):
        return {
            "distance": np.sqrt(np.square(self.fingertip_position[0]-self.target_position[0]) + np.square(self.fingertip_position[1]-self.target_position[1]) + 0) #Z axis distance is 0 for now as we are working with a 2D world right now
        }
    #Method to move base along the horizontal axis
    def move_base(self):
        self.base_position = (self.base_position[0] + self.base_vel, self.base_position[1]) #Increment the base position's x coordinate by the velocity

    #Moving the target along the sunusoidal path
    def move_target(self):
        # Update the target's x-coordinate linearly
        x_position = self.target_position[0] + self.target_x_speed

        # Update the target's y-coordinate based on a sinusoidal curve
        y_position = self.window_size/2-(self.target_amplitude * np.sin(self.target_frequency * self.time_step))

        # Update the target position
        self.target_position = (x_position, y_position)

        # Increment the time step
        self.time_step += 1

    def _calculate_arm_positions(self):
        # Calculate the position of the first joint
        joint1_x = self.base_position[0] + self.arm_length1 * np.cos(self.joint0)
        joint1_y = self.base_position[1] + self.arm_length1 * np.sin(self.joint0)

        # Calculate the position of the fingertip
        fingertip_x = joint1_x + self.arm_length2 * np.cos(self.joint1)
        fingertip_y = joint1_y + self.arm_length2 * np.sin(self.joint1)

        return (joint1_x, joint1_y), (fingertip_x, fingertip_y)
    
    # This is called at initialization and at close. According to the OpenAI documentation, we can assume reset is called before step is called. 
    def reset(self):

        self.base_position = (0,self.window_size/2) #Starts from the left side of the screen
        self.fingertip_position = (0,0)
        self.joint0 = np.pi/2 #Joint0 angle
        self.joint1 = np.pi/3 #Joint1 angle
        self.vel0 = 0 # Angular velocity of arm 1
        self.vel1 = 0 # Angular velocity of arm 2 
        #TODO: Need to make this random value within area of circle formed by radius =  2 x size
        self.fingertip_position = (0.5, 0.5)
        self.target_position = (0, self.window_size/2 - 50) #Start from the left of the screen moving horizontally along a sinusoidal path 

        self.base_vel = 5 #Speed at which the base is moving horizontally

        self.target_x_speed = 5 #speed of target along the x axis

        # Parameters for the sinusoidal movement of the target
        self.target_amplitude = 100  # Amplitude of the sinusoidal curve
        self.target_frequency = 0.1    # Frequency of the sinusoidal curve
        self.time_step = 0           # Time step counter

        observation = self.__get_obs()
        info = self.__get_info()
        return observation, info
    
    # This is the method that updates the state of our environment at each timestamp
    #TODO: Implement method to return observation, info, terminated, truncated, and reward
    def step(self, action):
        #TODO: Apply random torques as action and update the obstervation
        # Constants for inertia and damping (these values are just examples)
        inertia = 1.0
        damping = 0.1

        # Update angular velocities based on applied torques and physics
        self.vel0 += (action[0] / inertia) - (damping * self.vel0)
        self.vel1 += (action[1] / inertia) - (damping * self.vel1)

        # Update joint angles based on new velocities
        self.joint0 += self.vel0
        self.joint1 += self.vel1

        # Keep joint angles within a reasonable range
        self.joint0 = self.joint0 % (2 * np.pi)
        self.joint1 = self.joint1 % (2 * np.pi)

        self.move_base() # Move base to the (x + speed, y position)
        self.move_target() # Move target to the next spot on the sinusoidal path
        obs = self.__get_obs() # Get updated observation
        info = self.__get_info() # Get updated info
        reward = 1/info["distance"] #TODO: Need to implement a reward function
        if self.render_mode == "human":
            self._render_frame() # Draw the scene onto the pygame window
        return (obs, reward, False, False, info) # Observation, reward, Terminated, Truncated, Info

    #TODO: Implement rendering
    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    #TODO: change skeleton code to fit our needs
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                self.close()


        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # First we draw the target
        pygame.draw.circle(canvas, (255,0,0), self.target_position, 5) #Target
        # Now we draw the agent
        pygame.draw.circle(canvas, (0,255,0),self.base_position, 5) #Base joint

        # Calculate arm positions
        joint1_pos, fingertip_pos = self._calculate_arm_positions()

        # Draw the arm segments
        pygame.draw.line(canvas, (0, 0, 255), self.base_position, joint1_pos, 2)  # First arm segment
        pygame.draw.line(canvas, (0, 0, 255), joint1_pos, fingertip_pos, 2)      # Second arm segment

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

##Testing
            
env_test = MovingReacher(render_mode="human")
assert env_test is not None, "Failed to create modified env"

initial_observation, _ = env_test.reset()
assert initial_observation is not None, "Reset method should return an initial observation"

action = env_test.action_space.sample()  # Generate a random action
observation, reward, terminated, truncated, info = env_test.step(action)
assert observation is not None, "Step method should return an observation"
assert isinstance(reward, float), "Step method should return a reward as float"
assert isinstance(terminated, bool), "Step method should return a 'done' flag as bool"
assert isinstance(truncated, bool), "Step method should return a 'done' flag as bool"
assert isinstance(info, dict), "Step method should return 'info' as a dictionary"

try:
    env_test.render()
except Exception as e:
    assert False, f"Rendering failed with an exception: {e}"

for episode in range(10):  # Test for 10 episodes
    observation = env_test.reset()
    done = False
    while not done:
        action = env_test.action_space.sample()
        observation, reward, terminated, truncated, info = env_test.step(action)
        env_test.render()
env_test.close()
