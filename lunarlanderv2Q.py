import numpy as np
import keras
import gym
import os
import h5py
import random
from collections import deque
from keras.models import Sequential
from keras import optimizers
from keras import layers
from keras import activations

# four possible actions: nothing, shoot blaster left right or down
numActions = 4
# eight variables: pretty sure its x pos, y pos, velocity vectors,
# rotational degree etc
numVariables = 8
# learning rate 
lRate = 0.001
# bellman discount
discount = 0.99
# maximum number of memories stored 400k since we are storing so many memories and i just never want it to overflow
maxMemory = 400000
# memory buffer
replayMemory = deque(maxlen=maxMemory)
# chance that rather than using the DQN, you take a random action and explore
exploreProb = 1.00
exploreProbMin = 0.05
decay = 0.995
# training epochs
trainingEpochs = 1
# sentinal values for some loops later
observeAndTrain = True
# number of games of lunar lander to actually play
numGames = 1000
# counter 
counter = 0


# make the environment
env = gym.make('LunarLander-v2')
env.reset()
numObsSpace = env.observation_space.shape[0]

# time to make the neural network 
model = Sequential()
# input layer
# we're using rectified linear units for the activation
# not gonna pretend i know exactly what that means but again, seems standard
model.add(layers.Dense(512, activation='relu', input_dim=numObsSpace))
# we're going to have one hidden layer
model.add(layers.Dense(256, activation='relu'))
# output layer with 4 nodes for the 4 possible actions
model.add(layers.Dense(numActions, activation=activations.linear))

# using the adam optimizer
optimizer = optimizers.Adam(lr=lRate)
# now compile the model
# using mean squared error 
model.compile(loss='mse', optimizer=optimizer)


outputFile = open("scores.txt", "w")

# choose either a random action or an action dictated by our network
def findAction(qstate):
     # with probability exploreProb, take a random action
    prob = np.random.rand(1)
    
    if(prob < exploreProb):
        # be random
        a=env.action_space.sample()
    else:
        # get complicated
        # effectively we want a = argmax(a'Q(s,a'))
        # we are gonna get that by predicting the total rewards for each state
        # via our network
        predictedActions = model.predict(qstate)
        a = np.argmax(predictedActions[0])
    return a

# add a memory to the array
def addToMemory(state, action, reward, nextState, done):
    # add it as an array
    replayMemory.append([state, action, reward, nextState, done])

# get the individual attributes from the memory
def getAttributes(sample):
    # extract the attributes from the sample memories
    states = []
    actions = []
    rewards = []
    nextStates = []
    doneList = []
    for i in sample:
          states.append(i[0])
          actions.append(i[1])
          rewards.append(i[2])
          nextStates.append(i[3])
          doneList.append(i[4])
    # make them np arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    nextStates = np.array(nextStates)
    doneList = np.array(doneList)
    # the states array is of a strange size, squeeze it twice to make it fit with thes rest of our stuff
    states = np.squeeze(states)
    states = np.squeeze(states)
    # likewise with nextStates
    nextStates = np.squeeze(nextStates)

    return states, actions, rewards, nextStates, doneList


# the algorithm i'm trying to follow:
# initialize replay memory R
# initialize action-value function Q (with random weights)
# observe initial state s
# repeat
# 	select an action a
# 		with probability ϵ select a random action
# 		otherwise select a= argmaxa′Q(s,a′)
# 	carry out action a
# 	observe reward rr and new state s’
# 	store experience <s,a,r,s> in replay memory R
# 	sample random transitions <ss,aa,rr,ss′>from replay memory R
# 	calculate target for each minibatch transition
# 		if ss’ is terminal state then tt =rr otherwise tt =rr + γmaxa′Q(ss′,aa′)
# 	train the Q network using (tt−Q(ss,aa))2 as loss
# 	s=s′
# until terminated

# this if lets you just tell the agent to play the game rather than learn
if observeAndTrain:
    for game in range(0, numGames):
        # get initial state
        qstate = env.reset()
        # the states need to be resized a bit to work with our other stuff
        qstate = np.reshape(qstate, [1, numObsSpace])
        # now step through this game
        episodeReward = 0
        for step in range(1000):
            # according to the OG DQN paper, we want to take 
            # a few random steps before moving to actual play
            if game < 15:
                a = env.action_space.sample()
            else:
                a = findAction(qstate)

            # if you don't want the graphic to render, comment this out
            env.render()

            # now we actually perform the action
            # step returns the next state, the reward, a sentinal for
            # if we are done, and some miscellaneous info that isn't relevant
            s, reward, done, info = env.step(a)
            s = np.reshape(s, [1,numObsSpace])
            # save the memory
            addToMemory(qstate, a, reward, s, done)
            # add the reward to our overall reward
            episodeReward += reward
            # assign the new state 
            qstate = s
            # keep a counter to fit every 5th step. fitting every step is excessive, fitting every tenth step had slower progress
            counter += 1
            counter = counter % 5
            
            # only do this if we have enough in memory and it is a fifth step
            if(len(replayMemory) > 64 and counter == 0):
                # take the sample of size 64
                randomSample = random.sample(replayMemory, 64)
                # get the attributes from that sample
                states, actions, rewards, nextStates, doneList = getAttributes(randomSample)
                # find the target from the sample of nextStates
                targets = rewards + discount * (np.amax(model.predict_on_batch(nextStates), axis=1)) * (1-doneList)
                # get predictions for what the model would do on previous states
                targetVals = model.predict_on_batch(states)
                # this is just for 0....63 for the targetVals
                indexes = np.array([i for i in range(64)])
                # assign the target values to be fitted
                targetVals[[indexes], [actions]] = targets
                # fit the model with 1 epoch
                model.fit(states, targetVals, epochs=1, verbose=0)
        
            # at the end of every fifth game print out some tracking info to the console
            if(done and game % 5 == 0):
                print("After Training game number ", game, " steps taken = ", step," last reward ", reward, " final score ", episodeReward)
        
            # we want to decay the rate by which we explore. exploring is good in the beginning when our model is bad
            # but exploring later on when the model is good is detrimental
            if exploreProb > exploreProbMin:
                exploreProb = exploreProb * decay
                    
            # if the game did end we need to leave
            if done:
                if reward > 50:
                    print("Game ", game," was won")
                # output the cumulative reward to a file for later graphing
                outputFile.write(str(episodeReward)+ "\n")
                break

# close the file in the end
outputFile.close()
