corpus = 'The earth revolves around the sun.'
import string
import numpy as np

def preprocessing(corpus):
    training_data = []
    sentences = corpus.split(".")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence]
        x = [word.lower() for word in x]
        training_data.append(x)
    return training_data
    
training_data = preprocessing(corpus=corpus)

def prepare_data_for_training(sentences):
    data = {}
    X = []
    y = []
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
      
    #for i in range(len(words)):
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
             
            for j in range(i-3,i+3):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab[sentence[j]]] += 1
            X.append(center_word)
            y.append(context)
    initialize(V,data)
  
    return X,y 

def initialize(V,data):
    N = 10
    word_index = {}
    W = np.random.uniform(-0.8, 0.8, (V, N))
    W1 = np.random.uniform(-0.8, 0.8, (N, V))
          
    words = data
    for i in range(len(data)):
        word_index[data[i]] = i

prepare_data_for_training(training_data)



