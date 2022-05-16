import pandas as pd
import numpy as np
from tqdm import tqdm

class Word2vec(object):
    def __init__(self, window_size, embedding_dimension) -> None:
        self.window_size = window_size
        self.embedding_dimension = embedding_dimension

        self.data = None
        self.word_to_id = None
        self.id_to_word = None
        self.vocab_size = None

    def fit(self, path_to_data):
        df = pd.read_csv(path_to_data, index_col=False)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        data = df.unstack().values

        self.data = data
        self._generate_word_to_id()
        self._generate_id_to_word()
        self.vocab_size = len(self.word_to_id)

    def _generate_word_to_id(self):
        word_to_id = dict()
        cnt = 0
        for review in self.data:
            for char in review.split(' '):
                if char not in word_to_id:
                    word_to_id[char] = cnt
                    cnt += 1
        self.word_to_id = word_to_id

    def _generate_id_to_word(self):
        self.id_to_word = {str(v):k for k,v in self.word_to_id.items()}

    def _get_low_context(self, idx, review):
        low_idx = np.max([0, idx-self.window_size])
        return review[low_idx:idx]

    def _get_high_context(self, idx, review):
        high_idx = np.min([len(review), idx+self.window_size])
        return review[idx:high_idx]

    def _concat_context(self, low, high):
        return low + high

    def _one_hot_encode_word(self, word):
        encoded = np.zeros(self.vocab_size, dtype=np.int)
        word_id = self.word_to_id[word]
        encoded[word_id] = 1
        return encoded

    def _one_hot_encode_context(self, context):
        encoded_context = np.zeros(self.vocab_size, dtype=np.int)
        for word in context:
            encoded_context[self.word_to_id[word]] = 1
        return encoded_context


    def generate_training_data(self, n=-1):
        X = []
        y = []
        reviews = self.data
        for k, review in tqdm(enumerate(reviews), desc='Generating training set'):
            if k == n:
                break
            else:
                words_review = review.split(' ')
                if '' in words_review:
                    words_review.remove('')
                elif ' ' in words_review:
                    words_review.remove(' ')
                for i, word in enumerate(words_review):

                    mid_word = word

                    low_context = self._get_low_context(idx=i, review=words_review)
                    high_context = self._get_high_context(idx=i+1, review=words_review)
                    context = self._concat_context(low_context, high_context)
                    
                    encoded_mid_word = self._one_hot_encode_word(word=mid_word)
                    encoded_context = self._one_hot_encode_context(context=context)

                    X.append(encoded_mid_word)
                    y.append(encoded_context)
                        
        return np.array(X), np.array(y)

    def get_word_with_id(self, id):
        return self.id_to_word[str(id)]




