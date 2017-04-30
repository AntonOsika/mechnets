import numpy as np
import tensorflow as tf
import gym

#Use tensorboard to see how parameters diverge -> logprob > 300



def RenderCartpole():
    env.reset()
    for i in range (250):
        probs = np.exp(tfLogpiSingleState.eval(feed_dict={tfSingleState:np.array(env.state).reshape([1, -1])})).reshape(-1)
        probs /= probs.sum()
        action = np.random.choice([0, 1], p=probs)
    observation, reward, done, info = env.step(action)
    print("probs: (%g, %g), taken: %d" % (probs[0],probs[1],action))
    env.render()
    if done:
        env.reset()
    env.reset()

    #Performance()



def Performance():
    env.reset()
    sumReward = 0.0
    sumRewards = []
    for i in range(nBatches): 
        action = session.run([tfAStar], feed_dict={tfSingleState:env.state})
        observation, reward, done, info = env.step(action)
        sumReward += reward
        if done:
            env.reset()
            sumRewards.append(sumReward)
            sumReward = 0.0

    perc = np.percentile(sumRewards, [10, 90])
    print "Average reward with perc: %g (%g, %g)" % (np.mean(sumRewards), perc[0], perc[1] )
    


env = gym.make('CartPole-v0')

batchSize = 128
learnRate = 1.0e-2
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n

graph = tf.Graph()
with graph.as_default():

    tfState = tf.placeholder( tf.float32, shape=(batchSize, stateDim), name="state" )
    tfValue = tf.placeholder( tf.float32, shape=(batchSize), name="value" )
    tfAction = tf.placeholder( tf.float32, shape=(batchSize, actionDim), name="action" )
    
    #tfDiscount = tf.constant( 1.0 )
    #tfOnes = tf.constant( 1.0, dtype=tf.float32, shape=(batchSize, 1) )

    N1 = 4
    N2 = 5

    W1 = tf.Variable( tf.random_normal( [stateDim, N2], stddev=0.1))
    b1 = tf.Variable( tf.zeros( [1, N2]) )

    W2 = tf.Variable( tf.random_normal( [N1, N2], stddev=0.1))
    b2 = tf.Variable( tf.zeros( [1, N2]) )

    W3 = tf.Variable( tf.random_normal( [N2, actionDim], stddev=0.1))
    b3 = tf.Variable( tf.zeros( [1, actionDim]) )
    
    def logpi(x):
        q = tf.matmul(x, W1) + b1
        q = tf.sigmoid(q)
        q = tf.matmul(q, W3) + b3
        return q

    tfLogpi = logpi( tfState )
    tfLogSumPi = tf.log( tf.reduce_sum( tf.exp( tfLogpi ), [1] ) )
    tfLoss = - tf.matmul( tf.reshape( tfValue, (1, -1) ), tf.reshape( tf.reduce_sum( tfLogpi * tfAction, [1] ) - tfLogSumPi, (-1, 1) ) )
    optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(tfLoss)

    
    tfSingleState = tf.placeholder( tf.float32, shape=(1, stateDim), name="singleState" )
    tfLogpiSingleState = logpi(tfSingleState)
    
nBatches = 1024
discountRate = 0.97
epsilon = 0.0

states = np.zeros((2*batchSize, stateDim))
values = np.zeros((2*batchSize))
actions = np.zeros((2*batchSize, actionDim))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    
    RenderCartpole()

    print "Initialized"
    epochStart = 0
    consumedSteps = 0
    for step in xrange(nBatches*batchSize):
        
        idx = step % states.shape[0]

        states[idx] = (env.state)
        probs = np.exp(tfLogpiSingleState.eval(feed_dict={tfSingleState:np.array(env.state).reshape([1, -1])})).reshape(-1)
        probs *= (1 - epsilon)/probs.sum()
        probs += epsilon/actionDim
        action = np.random.choice([0, 1], p=probs)
        observation, reward, done, info = env.step(action)
        actions[idx, action] = 1.0
        actions[idx, 1-action] = 0.0
        values[idx] = reward

        #print("probs: ", probs, " action: ", action, " actions: ", actions[idx], " values ", values[idx], " states: ", states[idx, :], " done: ", done)
        #raw_input("")

        if done or step - epochStart == nBatches:
            for i in range(idx-1, epochStart, -1):
                values[i-1 % states.shape[0]] += values[i % states.shape[0]]*discountRate
                #print values[i-1 % states.shape[0]], i
            env.reset()
            epochStart = step

            if step - consumedSteps >= batchSize:
                idx1 = consumedSteps % states.shape[0]
                idx2 = idx1 + batchSize
                feed_dict = {tfState:states[idx1:idx2, :]
                    , tfValue:values[idx1:idx2]
                    , tfAction:actions[idx1:idx2].astype(np.bool) }

                _, loss = session.run([optimizer, tfLoss], feed_dict=feed_dict)
                consumedSteps += batchSize
        
                print "Loss at step %d %g, probs: (%g, %g)" % (step, loss[0], probs[0], probs[1])

    RenderCartpole()

