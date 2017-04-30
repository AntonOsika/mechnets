import tensorflow as tf
import numpy as np

batchsize = 128
learnrate = 1e-8
N1 = 3

graph = tf.Graph()
with graph.as_default():
    ab_tf = tf.placeholder(tf.float32, shape=(batchsize, 2))
    y_tf = tf.placeholder(tf.float32, shape=(batchsize, 1))

    w1 = tf.Variable(tf.truncated_normal([2, N1]))
    b1 = tf.Variable(tf.zeros([N1]))
    w2 = tf.Variable(tf.truncated_normal([N1, 1]))
    b2 = tf.Variable(tf.zeros([1]))

    l1 = tf.nn.sigmoid(tf.matmul(ab_tf, w1) + b1)
    ypred = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)

    loss_tf = tf.reduce_mean(tf.square(y_tf - ypred))
    optimizer = tf.train.GradientDescentOptimizer(learnrate).minimize(loss_tf)
    

steps = 10001
with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()

    for step in xrange(steps):
        x = np.random.uniform(size=(batchsize, 2))
        y = (x[:,0]*x[:,1]).reshape((128,1))

        loss, _ = session.run([loss_tf, optimizer], feed_dict={ ab_tf:x, y_tf:y})

        if step % 100 == 0:
            print "Loss: %g" % (loss)

            x = 2*np.random.uniform(size=(batchsize, 2))
            y = (x[:,0]*x[:,1]).reshape((128,1))
            loss = session.run([loss_tf], feed_dict={ ab_tf:x, y_tf:y})[0]

            print "Loss, test set: %g" % (loss)

