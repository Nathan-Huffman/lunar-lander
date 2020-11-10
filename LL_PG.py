'''
LunarLander_PolicyGradient
author: Nathan Huffman
version: 0.1

This library implements Policy Gradient reinforcement learning, namely for Lunar Lander.
It Also contains wrappers for running this agent on any given environment.
'''


#---Parameters-----------------
save_dir = 'checkpnts'
save_file = None    # 'Lunar_16_24_16'
restore_file = None # 'LunarLander-v2_(256, 256)-0.001-0.95_ 200'
#------------------------------
episodes = 2500
learning_rate = 0.0075
discount = 0.995
layers = (16,24,16,8)
target_score = 200
target_window = 100
save_interval = 500
#------------------------------
step_limit = 500
step_limit_penalty = -10
#------------------------------

# Allow CPU to be preferred with '-c' or '--cpu' command line arg
from sys import argv
from os import path, makedirs, environ
if '-c' in argv or '--cpu' in argv:
    environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import required modules (tensorflow must be after CUDA config above)
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import GradientTape, squeeze, convert_to_tensor
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions

# Reinforcement learning agent that utilizing Policy Gradient
class PolicyGradient:
    def __init__(self, n_inputs, n_actions, alpha=learning_rate,
                 gamma=learning_rate, layers=layers, restore_file=None):
        self.gamma = gamma          # Store discount factor
        self.n_inputs = n_inputs    # Store dimension of input
        self.n_actions = n_actions  # Store dimension of output
        self.action_space = [i for i in range(self.n_actions)]
        self.clear_steps()          # Initialize step memory
        
        if restore_file:    # If model file specified, load it
            self.policy = load_model(restore_file)
        else:               # Otherwise, build madel based on layer and learning rate specs
            self.build_model(layers, alpha)
        
    # Create neural network based on layer and learning rate specifications
    def build_model(self, layers, lr):
        state = Input(shape=(self.n_inputs,))   # Create inputs to take in state info 
        build = state
        for layer in layers:                    # Build specified dense hidden layers
            build = Dense(layer, activation='relu')(build)
        probs = Dense(self.n_actions,           # Add output layer, with distributed action probabilites
                      activation='softmax',dtype='float64')(build)
        
        self.policy = Model(inputs=[state], outputs=[probs])    # Build model with inputs and outputs
        self.policy.compile(optimizer=Adam(lr=lr))              # Compile using Adam optimization
    
    # Use the policy to determine which action to take
    def select_action(self, state):
        state = convert_to_tensor([state], dtype='float32')     # Add efficiency to calc
        probs = self.policy(state,training=False)[0]            # Get weighted probabilities from agent
        action = np.random.choice(self.action_space, p=probs)   # Use bias to select action
        return action
        
    # Store data from each step in the environment
    def store_step(self, state, action, reward):
        self.state_memory.append(state)     # Store state presented to agent
        self.action_memory.append(action)   # Store action taken by agent
        self.reward_memory.append(reward)   # Sore reward earned by action
    
    # Clear stored memory of data from each step
    def clear_steps(self):
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
    
    # Calulcate propogated rewards of each step taken
    def calc_advantages(self):
        rewards = np.array(self.reward_memory)
        G = np.zeros_like(rewards)  # Create advantage array to make step rewards
        G_sum = 0                   # Calculate propogation of rewards
        for step in reversed(range(len(rewards))):
            G_sum += rewards[step]  # Add reward to running total
            G[step] = G_sum         # Save running total to step reward
            G_sum *= self.gamma     # Discount reward, controlling propogation
        
        std = np.std(G) if np.std(G) > 0 else 1
        return (G-np.mean(G))/std       # Standardize advantages
        
    # Improve own policy, using gradient ascent that utilizes model's certainty
    def learn(self):
        actions = np.array(self.action_memory)  # Get actions taken
        G = self.calc_advantages()              # Calc advantages of each action
        
        with GradientTape() as tape:            # Calculate gradients for all weights
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = convert_to_tensor([state], dtype='float32')     # Makes training faster
                probs = self.policy(state,training=False)[0]            # Get probability predictions based on each step state
                action_probs = distributions.Categorical(probs=probs)   # Create distribution based on probabilities
                log_prob = action_probs.log_prob(actions[idx])          # Find logistic probability of action being taken
                
                loss += -g * squeeze(log_prob)                          # Update loss based on reward and model's certainty
                
        # Calculate gradients from observing tape, then apply them to the policy weights
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        
        self.clear_steps()  # Clear step memory
       
# Encapsulates running of a specified agent on an environment
class EnvRunner:
    def __init__(self, env_name, agent_name, learning_rate=learning_rate, discount=discount,
                 layers=layers, target=None, target_window=100, restore_file=None, save_file=None):
        # Save the private fields
        self.score_memory = []
        self.target = target
        self.target_window = target_window
        self.episodes_trained = 0
        
        # Create the specified environment
        self.env = gym.make(env_name)
        n_inputs = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        
        # Retreive model from specified file
        if restore_file:
            restore_file, save_file = self.restore_prep(restore_file, save_file)
            self.agent = agent_name(n_inputs, n_actions, restore_file=restore_file)
        else:
            # Create the agent with normal parameters, creating the storage path that describes the model
            self.agent = agent_name(n_inputs, n_actions, alpha=learning_rate, gamma=discount, layers=layers)
            if not save_file:   # Create path name of where to save model file
                save_file = '{}_{}-{}-{}'.format(env_name, layers, learning_rate, discount).replace(' ','')
        
        self.save_file = save_file  # Save name of model
    
    # Format restore and save paths, as well as pull rewards from file
    def restore_prep(self, restore_file, save_file):
       # Extract info from the restore filename
        parse_restore = restore_file.rpartition('_',)
        self.episodes_trained, model_dir = int(parse_restore[-1]), parse_restore[0]
        # Connect the elements to form the path to the restore
        restore_file = path.join(save_dir,model_dir,restore_file)
        
        # Save reward memory to file
        reward_file = open('{}.txt'.format(restore_file), 'r')
        self.score_memory = [float(line.strip()) for line in reward_file]
        reward_file.close()
        
        if not save_file: save_file = model_dir         # Create default save_file if preferernce doesn't exist
        
        return '{}.tf'.format(restore_file), save_file  # Return formatted restore file and save_file
    
    # Export model and rewards to files
    def save_model(self, episode):
        model_dir = path.join(save_dir,self.save_file)    # Create model directory if not exist
        if not path.exists(model_dir): makedirs(model_dir)
        
        save_episode = path.join(model_dir,'{}_{:d}'.format(self.save_file,episode))
        save_model(self.agent.policy, '{}.tf'.format(save_episode,episode), overwrite=True, save_format='tf')
        
        reward_file = open('{}.txt'.format(save_episode,episode), 'w+')
        for line in self.score_memory:     # Write reward memory to file
            reward_file.write('{:.4f}\n'.format(line))
        reward_file.close()
    
    # Print training updates to the console
    def display_update(self, episode, steps, score, target=None, running_avg=None, target_window=100):
        print('Episode: {:4d} | Steps: {: >4d} | Score = {: >10}'.
                format(episode, steps, '{:.4f}'.format(score)), end=('' if running_avg else '\n'))
        if target:      # If flgged, also give info about the running average 
            print('   -   Past {}: {: >7}'.format(target_window,'{:.2f}'.format(running_avg)))
            if running_avg > target:
                print('Episode: {:4d} | Solved!                    {:>4} < {: >7}'.
                      format(episode, target, '{:.2f}'.format(running_avg)))
    
    # Save a graph of the learning progress
    def graph_learning(self):
        save_name = '{}_{:d}.png'.format(self.save_file, self.episodes_trained)
        save_loc = path.join(save_dir, self.save_file, save_name)
        plt.plot(self.score_memory, label='Agent Score')
        plt.xlabel('Episode',fontsize=14); plt.ylabel('Policy Score',fontsize=14)
        plt.title('Policy Gradient - Agent Learning',fontsize=18)
        plt.legend(loc='upper left')
        plt.ylim((-500, 300))
        plt.savefig(save_loc)
        plt.show()
    
    # Execute one episode using the policy
    def execute(self, render=False):    
        done = False; score = 0; n_steps = 0    # Reset episode variables
        state = self.env.reset()
        while not done and n_steps < step_limit:
            if render: self.env.render()                            # If flagged, show the environment     
            action = self.agent.select_action(state)                # Use policy to get next action
            state_next, reward, done, info = self.env.step(action)  # Apply action to env and get results
            self.agent.store_step(state, action, reward)            # Store results for learning
            
            state = state_next  # Update state to be current
            score += reward     # Update score and num steps
            n_steps += 1
        
        if n_steps >= step_limit:           # If agent takes to long to finish
            self.agent.reward_memory[-1] = step_limit_penalty
            score += step_limit_penalty     # Penalize model for taking too long
        
        return score, n_steps
    
    # Train the model by executing episodes and correcting the agent
    def train(self, episodes=1000, target=None, target_window=100):
        print('Starting training...')
        for episode in range(self.episodes_trained+1,self.episodes_trained+episodes+1):
            score, n_steps = self.execute()     # Run the model for one episode
            self.score_memory.append(score)    # Store the reward from the episode
            
            self.agent.learn()                  # Improve policy by executing agent's learning func
            
            # Calcualte running aversage and display updates
            running_avg = np.mean(self.score_memory[-target_window:])
            self.display_update(episode, n_steps, score, target, running_avg, target_window)

            self.episodes_trained = episode         # Update current training episode
            if episode%save_interval == 0:          # Reguarly save the model
                self.save_model(episode)
            if target and running_avg >= target:    # If reach target, display result
                break
                
        self.save_model(self.episodes_trained)      # Also save the model once finished with episodes
        print("Name: {}_{}".format(self.save_file, self.episodes_trained))
                
    # Display agent playing the environment
    def eval(self):
        self.graph_learning()
        while True:
            self.execute(render=True)

# Handle optional command line arguemnts and update variables
def handle_args():
    description = "Policy Gradient Library | Learns Lunar lander"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c","--cpu", help="Force to execute on CPU", action="store_true")
    parser.add_argument("-e","--eval", help="Only perform evaluation", action="store_true")
    parser.add_argument("-t","--train", help="Specify num training episdes", type=int)
    parser.add_argument("-l", "--layes", help="Specify number nodes for each layer", type=int)
    parser.add_argument("-a","-lr", "--learning_rate", help="Specify alpha / learning rate", type=float)
    parser.add_argument("-g","-d", "--discount", help="Specify gamma / discount rate", type=float)
    parser.add_argument("-lm","--limit", help="Specify step limit before terminating agent", type=int)
    parser.add_argument("-p","--penalty", help="Specify penalty for timing out", type=int)
    parser.add_argument("-r","--restore", help="Specify filename to restore from")
    parser.add_argument("-s","--save", help="Specify filename to save to")
    
    args = parser.parse_args()
    
    if args.train: global episodes; episodes = args.train
    if args.learning_rate: global learning_rate; learning_rate = args.learning_rate
    if args.discount: global  discount; discount = args.discount
    if args.limit: global step_limit; step_limit = args.limit
    if args.penalty: global step_limit_penalty; step_limit_penalty = args.penalty
    if args.restore: global restore_file; restore_file = args.restore
    if args.save: global save_file; save_file = args.save

    return args

# Main driver, learns Lunar Lander then evaualtes
if __name__ == '__main__':
    args = handle_args()
    
    runner = EnvRunner('LunarLander-v2',PolicyGradient, learning_rate, discount,
                       layers, target_score, target_window, restore_file, save_file)
    if not args.eval:
        runner.train(episodes, target_score)
    
    runner.eval()