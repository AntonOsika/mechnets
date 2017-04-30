
frames = 360
state = env.reset()
done = False
with tf.Session(graph=graph) as session:
    for frame in range(frames):
        env.render()
        if done:
            state = env.reset()

        action = session.run(tf_action, feed_dict={singlestate:state[np.newaxis, :]})

        state, reward, done, info = env.step(action)
        
