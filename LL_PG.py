'''
LunarLander_PolicyGradient
Nathan Huffman
'''

#---------------
save_dir = 'LL_PG_saves'
max_episodes = 5000
#---------------
alpha = 0.005
gamma = 0.9
layers = (256, 256)
#---------------

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time

import gym
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import GradientTape, squeeze, convert_to_tensor, float32
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions

class PolicyGradient(object):
    def __init__(self, n_inputs=8, n_actions=4, alpha=0.01, gamma=0.99, layers=(16,24,16)):
        self.gamma = gamma
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.action_space = [i for i in range(self.n_actions)]
        #self.save_path = os.path.join(dir)
        self.build_model(layers, alpha)
        
    def build_model(self, layers, lr):
        # Create state and reward inputs 
        state = Input(shape=(self.n_inputs,))
        # Build specified dense hidden layers
        build = state
        for layer in layers:
            build = Dense(layer, activation='relu')(build)
        # Add output layer, with probability distro
        probs = Dense(self.n_actions, activation='softmax',dtype='float64')(build)
        
        # Build and complie model with Adam optimization
        self.policy = Model(inputs=[state], outputs=[probs])
        self.policy.compile(optimizer=Adam(lr=lr))
    
    def select_action(self, state):
        #state = np.expand_dims(state, axis=0)
        #prob = self.policy(state)[0]
        state = convert_to_tensor([state], dtype='float32')
        probs = self.policy(state,training=False)[0]
        action = np.random.choice(self.action_space, p=probs)
        #print('Prob:{} / Action: {}'.format(prob, action))
        return action
        
    def store_step(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def clear_steps(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
    
    def calc_advantages(self):
        # Convert memory list to array
        rewards = np.array(self.reward_memory)
        G = np.zeros_like(rewards)
        # Calculate propogation of rewards
        G_sum = 0
        for step in reversed(range(len(rewards))):
            G_sum += rewards[step]
            G[step] = G_sum
            G_sum *= self.gamma
        # Standardize advantages
        std = np.std(G) if np.std(G) > 0 else 1
        return (G-np.mean(G))/std
        #return G 
        
    def actions_one_hot(self):
        # Convert memory list to numpy array
        actions = np.array(self.action_memory)
        # Transform into one-hot encoding
        actions_hot = np.zeros([len(actions), self.n_actions],dtype='int')
        actions_hot[np.arange(len(actions)),actions] = 1
        return np.float64(actions_hot)
    
    def learn(self):
        # Format step memory values
        #states = np.array(self.state_memory)
        actions = np.array(self.action_memory)
        G = self.calc_advantages(actions)
        
        # with GradientTape() as tape:
        #     loss = 0
        #     for idx, (g, state) in enumerate(zip(G, self.state_memory)):
        #         start_time=time.time()
        #         state = convert_to_tensor([state], dtype='float32')
        #         #state = np.expand_dims(state, axis=0)
        #         probs = self.policy(state,training=False)[0]
        #         #probs = K.clip(probs, 1e-8, 1-1e-8)
        #         #action_probs = distributions.Categorical(probs=probs)
        #         #log_prob = action_probs.log_prob(actions[idx])
        #         #time_distro = time.time()-start_time
        #         log_prob=np.log(probs[actions[idx]])

                
        #         #loss += -g * K.log(probs)
        #         loss += -g * log_prob
                
        #         time_raw = time.time()-start_time
        #         time_raw_sum += time_raw
        
        with GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = convert_to_tensor([state], dtype='float32')
                probs = self.policy(state,training=False)[0]
                action_probs = distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                
                loss += -g * squeeze(log_prob)
                
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        
        # Run training model on step memory
        #cost = self.predict.train_on_batch(states, G)
        # Clear step memory and return training cost
        self.clear_steps()
        #return cost
        
        
if __name__ == '__main__':
    agent = PolicyGradient(alpha=alpha, gamma=gamma, layers=layers)
    env = gym.make('LunarLander-v2')
    score_history = []
    
    for episode in range(max_episodes):
        # Execute policy
        done = False
        score = 0
        state = env.reset()
        while not done:
            # Show env
            #env.render()
            
            # Use policy to get next action
            action = agent.select_action(state)
            # Apply action to env and get results
            state_next, reward, done, info = env.step(action)
            # Store results for learning
            agent.store_step(state, action, reward)
            # Update state to be current
            state = state_next
            # Update score
            score += reward
        score_history.append(score)
        
        # Improve policy
        agent.learn()
        running_avg = np.mean(score_history[-100:])
        print('Episode: {:4d} | Score = {:9f}     Past 100:{:3f}'.format(episode, score, running_avg))
        
    agent.select_action()