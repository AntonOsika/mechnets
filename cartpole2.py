import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

batchSize = 128
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n

graph = tf.Graph()
with graph.as_default():

    tfState = tf.placeholder( tf.float32, shape=(batchSize, stateDim), name="state")
    tfReward = tf.placeholder( tf.float32, shape=(batchSize, 1), name="reward")
    tfQ1hat = tf.placeholder( tf.float32, shape=(batchSize, 1), name="Q1hat")
    tfAction = tf.placeholder(tf.int32, shape=(batchSize, 1), name="action")
    tfDone = tf.placeholder(tf.float32, shape=(batchSize, 1), name="done")
    
    tfDiscount = tf.constant( 1.0 )
    tfOnes = tf.constant( 1.0, dtype=tf.float32, shape=(batchSize, 1) )

    N1 = 4
    N2 = 2

    W1 = tf.Variable( tf.random_normal( [stateDim, N1], stddev=0.1))
    b1 = tf.Variable( tf.zeros( [1, N1]) )

    W2 = tf.Variable( tf.random_normal( [N1, N2], stddev=0.1))
    b2 = tf.Variable( tf.zeros( [1, N2]) )

    W3 = tf.Variable( tf.random_normal( [N2, actionDim], stddev=0.1))
    b3 = tf.Variable( tf.zeros( [1, actionDim]) )
    
    def Q(x):
        q = tf.matmul(x, W1) + b1
        q = tf.sigmoid(q)
        q = tf.matmul(q, W2) + b2
        q = tf.sigmoid(q)
        q = tf.matmul(q, W3) + b3
        return q

    tfQ0 = Q(tfState)
    tfQ0taken = tf.transpose(tf.gather(tf.transpose(tfQ0), tfAction))
    tfLoss = tf.nn.l2_loss( tfDiscount*tf.mul(tfOnes - tfDone, tfQ1hat) + tfReward - tfQ0 )
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(tfLoss)
    
    # TODO: Make trainable=False variable instead of calling twice.
    # This is used when getting frozen optimal Q for future
    tfQ0hat = tf.reduce_max(tfQ0, [1])


    # For using the network:
    tfSingleState = tf.placeholder( tf.float32, shape=( 1, stateDim ))
    tfAStar = tf.arg_max(Q(tfSingleState), 1)

    
nBatches = 1024
sumReward = 0.0

states0 = []
states1 = []
actions = []
rewards = []
dones = []

finalStates = []
sumRewards = []

i = 0
while i < batchSize*nBatches:
    states0.append(env.state)
    action = env.action_space.sample()
    actions.append(action)
    observation, reward, done, info = env.step(action)
    states1.append(observation)
    rewards.append(reward)
    dones.append(done)
    sumReward += reward

    if done:
        finalStates.append(observation)
        env.reset()
        sumRewards.append(sumReward)
        sumReward = 0.0
    i += 1

perc = np.percentile(sumRewards, [10, 90])
print "Average reward with perc: %g (%g, %g)" % (np.mean(sumRewards), perc[0], perc[1] )

states0 = np.array(states0)
states1 = np.array(states1)
rewards = np.array(rewards).reshape((-1, 1))
actions = np.array(actions).reshape((-1, 1))
dones = np.array(dones).reshape((-1, 1))

finalStates = np.array(finalStates)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print "Initialized"
    for step in xrange(nBatches*10):
        idx = step % states0.shape[0]

        feed_dict = {tfState:states1[idx:(idx+batchSize),:]}
        Qhat = tfQ0hat.eval(feed_dict=feed_dict)

        feed_dict = {tfState:states0[idx:(idx+batchSize),:]
                , tfAction:actions[idx:(idx+batchSize)]
                , tfReward:rewards[idx:(idx+batchSize)]
                , tfQ1hat:Qhat.reshape((-1,1))
                , tfDone:dones[idx:(idx+batchSize)]}

        _, loss = session.run([optimizer, tfLoss], feed_dict=feed_dict)
        
        if step % 100 == 0:
            print "Loss at step %d %g" % (step, loss)


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
    

    
