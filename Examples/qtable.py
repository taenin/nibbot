import gym
import numpy as np
import matplotlib.pyplot as plt

#load the environment
env = gym.make('FrozenLake-v0')
# Set learning params
#lr = learning rate
lr = 0.8
#y = retention of old info
y = .95
num_episodes = 2000

#Matt here: for fun, let's try tuning the learning rate:
lrMax = .9
lr = .1
lrIncr = .1
lrOutput = []
bestScore = []
while lr <= lrMax:
    #Initialize our q-table with all zeros (numpy.zeros)
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    #create lists to contain total reqards and steps per episode
    jList = []
    rList = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #Q-table learning algo:
        while j < 99:
            j+=1
            #Choose an action by greedily (with noise) picking from Q-table
            a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
            #Get new state and reward from our environment
            s1,r,d,_ = env.step(a)
            #Update the Q-table with new knowledge
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
            #Update the total reward
            rAll += r
            #update our state to the new state
            s = s1
            if d == True:
                break
        rList.append(rAll)
    lrOutput.append(lr)
    bestScore.append(sum(rList) / num_episodes)
    print("Score over time: " + str(sum(rList) / num_episodes))
    print("Final Q-table values")
    print(Q)
    lr += lrIncr
plt.plot(lrOutput, bestScore, 'bo')
plt.show()