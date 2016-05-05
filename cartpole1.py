import tensorflow as tf

state_dim = 4
action_dim = 1
out_dim = 2

N1 = 6
rate = 0.5
batchsize = 128

graph = tf.Graph()
with graph.as_default():
    startstate = tf.placeholder(tf.float32, shape=(batchsize, state_dim))
    actiontaken = tf.placeholder(tf.float32, shape=(batchsize, action_dim))
    Qtrue  = tf.placeholder(tf.int32, shape=(batchsize, 1))   # simple Selector for which Q to use

    weights1 = tf.Variable( tf.truncated_normal([state_dim, N1]))
    biases1 = tf.Variable(tf.zeros([N1]))

    # Only use the in_dim state variables 
    h1 = tf.nn.softmax(tf.matmul(startstate, weights1) + biases1)

    weights2 = tf.Variable( tf.random_normal([N1, out_dim]))
    biases2 = tf.Variable(tf.zeros([out_dim]))

    Qvals = tf.matmul(h1, weights2) + biases2
    Qpred = Qvals[:, actiontaken]]

    loss = tf.nn.l2_loss(Qpred-Qtrue)
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

    Qbest = tf.maximum(Qvals[:, 0], Qvals[:, 1])

    singlestate = tf.placeholder(tf.float32, shape(1,in_dim))
    action = tf.argmax( tf.matmul( tf.matmul(singlestate, w1) + biases1, w2) + biases2))

import gym

env = gym.make('CartPole-v0')
eps1 = 0.25
steps = 801

batches = 8
memsize = batchsize*batches
states1 = np.zeros((memsize, in_dim +1))     #With action taken
states2 = np.zeros((memsize, in_dim + 1))    #With reward
pos = 0
done = false
state = env.reset()

with tf.Session(graph=graph) as session:
    for step in range(steps):
        eps = (1 - step/(steps-1))*eps1
        for pos in range(batchsize):
            for i in range( batches ):
                if done:
                    state = env.reset()
                if np.random.rand() < eps:
                    action = np.random.randint() % 2
                else:
                    action = session.run(action, feed_dict={ singlestate:state })

                states1[pos +i*batchsize, :in_dim] = state
                states1[pos +i*batchsize, in_dim] = action

                observation, reward, done, info = env.step(action)

                if done:
                    reward += -10

                states2[pos +i*batchsize, :in_dim] = state
                states2[pos +i*batchsize, in_dim] = reward
        for batch in range(batches):
            # Future max Q + reward:
            offset = batch*batchsize
            Qtrue = session.run(Qbest, feed_dict={ startstate:states2[offset:(offset+batchsize),:state_dim] }) + states2[offset:(offset+batchsize), state_dim]
            _, l = session.run([optimizer, loss], feed_dict={startstate:states1[offset:(offset+batchsize), :state_dim], actiontaken:states1[offset:(offset+batchsize), -1], Qtrue:Qtrue}
            if step % 100 == 0:
                print("Loss at step", step, ":", l)
                print("Avg Q:", Qtrue.mean())
