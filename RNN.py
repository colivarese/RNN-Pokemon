import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tqdm import tqdm

class RNN(object):
    def __init__(self, chunk_size=6380, n_chunks=1, rnn_size=128, num_out=6380, epochs=50, batch_size=128):
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.rnn_size = rnn_size
        self.num_out = num_out
        self.optimizer = tf.train.AdamOptimizer(1e-2)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self):
        self.reset_graph()

        
        x_in = tf.placeholder('float', [None, self.n_chunks, self.chunk_size ])
        y = tf.placeholder('float')

        x = tf.transpose(x_in, [1,0,2])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x, self.n_chunks, 0)

        lstm_cell=tf.contrib.rnn.LSTMCell(self.rnn_size,state_is_tuple=True,num_proj=self.num_out)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        cost = tf.losses.mean_squared_error(y,outputs[ self.n_chunks-1]) 
        self.optimizer = self.optimizer.minimize(cost)
        
        return dict(
              x=x_in,
              y=y,
              output=outputs[self.n_chunks-1],
              states=states,
              cost=cost,
              optimizer=self.optimizer,
              saver = tf.train.Saver()  
              )

    def reset_graph(self):
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()    

    def train(self, model, train_x, train_y, save='RNN'):

        p = np.random.permutation(len(train_x))
        train_x, train_y = train_x[p], train_y[p]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in tqdm(range(self.epochs)):
                epoch_loss = 0
                for i in range(int(len(train_x)/self.batch_size)):
                    epoch_x = train_x[self.batch_size*i:self.batch_size*(i+1)]
                    epoch_y = train_y[self.batch_size*i:self.batch_size*(i+1)]
                    epoch_x = epoch_x.reshape((self.batch_size, self.n_chunks, self.chunk_size))
                    feed_dict = {model['x']:np.array(epoch_x), 
                                model['y']: np.array(epoch_y)}
                    h, c, prediction, y = sess.run([model['optimizer'],
                                                    model['cost'],
                                                    model['output'],
                                                    model['y']],
                                                    feed_dict=feed_dict)
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of',self.epochs,'loss:',epoch_loss)

            if isinstance(save, str):
                ENCname="./SVE/"+save+".ckpt"
                model['saver'].save(sess, ENCname)

    


    