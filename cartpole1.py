import tensorflow as tf
import numpy as np

state_dim = 4
#action_dim = 1
out_dim = 2

N1 = 3
N1 = 3
rate = 0.5
discount = 0.5
batchsize = 128
batches = 8
memsize = batchsize*batches

graph = tf.Graph()
with graph.as_default():
    startstate = tf.placeholder(tf.float32, shape=(batchsize, state_dim))
    actiontaken = tf.placeholder(tf.bool, shape=(batchsize))
    Qtrue  = tf.placeholder(tf.float32, shape=(batchsize))   # simple Selector for which Q to use

    weights1 = tf.Variable( tf.truncated_normal([state_dim, N1]))
    biases1 = tf.Variable(tf.zeros([N1]))

    # Only use the state_dim state variables 
    h1 = tf.nn.sigmoid(tf.matmul(startstate, weights1) + biases1)

    weights2 = tf.Variable( tf.random_normal([N1, out_dim]))
    biases2 = tf.Variable(tf.zeros([out_dim]))

    Qvals = tf.matmul(h1, weights2) + biases2
    Qpred = tf.select(actiontaken, Qvals[:, 0], Qvals[:, 1])

    loss = tf.nn.l2_loss(Qpred-Qtrue)
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

    Qbest = tf.maximum(Qvals[:, 0], Qvals[:, 1])

    singlestate = tf.placeholder(tf.float32, shape=(1,state_dim))
    tf_action = tf.argmax( tf.matmul( tf.nn.sigmoid(tf.matmul(singlestate, weights1) + biases1), weights2) + biases2, 1)

    startstate_test = tf.placeholder(tf.float32, shape=(memsize, state_dim))
    Qvals_test = tf.matmul( tf.nn.sigmoid(tf.matmul(startstate_test, weights1) + biases1), weights2) + biases2
    Qbest_test = tf.maximum(Qvals_test[:, 0], Qvals_test[:, 1])

import gym

env = gym.make('CartPole-v0')
eps1 = 0.25
steps = 801

states1 = np.zeros((memsize, state_dim ))     #With action taken
actions = np.zeros((memsize), dtype='bool_')     #With action taken
states2 = np.zeros((memsize, state_dim + 1))    #With reward
pos = 0
done = False
state = env.reset()
l = 0
counter = 0
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in xrange(steps):
        eps = (1 - step/(steps-1))*eps1
        if step % 100 == 0:
            for pos in xrange(batchsize):
                for i in xrange( batches ):
                    if done:
                        state = env.reset()
                    if np.random.rand() < eps:
                        action = np.random.randint(0, 2)
                    else:
                        action = session.run(tf_action, feed_dict={ singlestate:state[np.newaxis, :] })

                    states1[pos +i*batchsize, :] = state
                    actions[pos +i*batchsize] = action

                    state, reward, done, info = env.step(action)

                    if done:
                        reward += -10

                    states2[pos +i*batchsize, :state_dim] = state
                    states2[pos +i*batchsize, state_dim] = reward

                    if step +1 == steps:
                        env.render()
                        if done:
                            counter += 1
                            print("died ", counter)

        if step % 100 == 0:
            # Average actual Q over all data
            Qhat = session.run([Qbest_test], feed_dict={ startstate_test:states2[:,:state_dim] }) + states2[:, state_dim]
            print("Loss at step", step, ":", l)
            print("Avg Q:", Qhat.mean())

        for batch in xrange(batches):
            # Future max Q + reward:
            offset = batch*batchsize
            Qhat = session.run(Qbest, feed_dict={ startstate:states2[offset:(offset+batchsize),:state_dim] }) + discount*states2[offset:(offset+batchsize), state_dim]
            _, l = session.run([optimizer, loss], feed_dict={startstate:states1[offset:(offset+batchsize), :state_dim], actiontaken:states1[offset:(offset+batchsize), -1], Qtrue:Qhat})

        if (step + 1) % 100 == 0:
            # Average actual Q over all data
            Qhat = session.run([Qbest_test], feed_dict={ startstate_test:states2[:,:state_dim] }) + states2[:, state_dim]
            print("Loss at step", step, ":", l)
            print("Avg Q:", Qhat.mean())
