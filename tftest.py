import tensorflow as tf
import numpy as np

graph = tf.Graph()
with graph.as_default():
    qq = tf.placeholder(tf.float32)
    ww = qq*2

with tf.Session(graph=graph) as session:
    w = session.run(ww, feed_dict={qq:np.random.rand()})
    print(w)

