import tensorflow as tf

in_dim = 4
out_dim = 2

N1 = 6

graph = tf.Graph()
with graph.as_default():
    input = tf.placeholder(tf.float32, shape=(batch_size, in_dim))
    output = tf.placeholder(tf.float32, shape=(batch_size, out_dim))

    weights1 = tf.Variable( tf.truncated_normal([in_dim, N1]))
    biases1 = tf.Variable(tf.zeros([N1]))

    h1 = tf.nn.relu(tf.matmul(input, weights1) + biases1)

    weights2 = tf.Variable( tf.random_normal([N1, out_dim]))
    biases2 = tf.Variable(tf.zeros([out_dim]))

    pred = tf.matmul(h1, weights2) + biases2

    loss = tf.nn.l2_loss(pred-output)
