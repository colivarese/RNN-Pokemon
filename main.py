from word2vec import Word2vec
from RNN import RNN
import random 
import tensorflow as tf
import numpy as np
import sys

w2v = Word2vec(window_size=2, embedding_dimension=150)

print(tf.__version__)
print(np.__version__)
print(sys.version)



#corpus = "The earth revolves around the sun. The moon revolves around the earth"

w2v.fit(path_to_data='cleaned_data.csv')

X, y = w2v.generate_training_data()

p = np.random.permutation(len(X))
X, y = X[p], y[p]

a = 5

rnn = RNN()
model = rnn.fit()
save = 'RNN'
rnn.train(model, X, y, save)


def predict(model, data, target, save):
        with tf.Session() as sess:
            ENCname = './SVE/'+save+'.ckpt'
            model['saver'].restore(sess, ENCname)
            output = []
            for i in range(len(data)):

                epoch_x = data[i].reshape((1,1,6380))
                prediction,y,h = sess.run([model['output'],
                                           model['y'],
                                           model['states']],
                                           feed_dict = {
                                               model['x']:epoch_x,
                                               model['y']:target[i]
                                           })
                output.append(prediction)
        return np.array(output), h

predictor_rnn = RNN()
predictor_model = predictor_rnn.fit()

N = 10
trained_model, hidden = predict(predictor_model, X, y, save)

random_sample = random.sample(range(1, 6380), 20)

for i in random_sample:
    pred = trained_model[i][0]
    idx = (-pred).argsort()[:5]
    word_id = np.argmax(X[i])
    word= w2v.get_word_with_id(word_id)
    print('Word-> {', word,'}', end =" ")
    print('Closests -> {', end=' ')
    for j in idx:
        print(w2v.get_word_with_id(j), ', ', end='')
    print('}')

