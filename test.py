from word2vec import Word2vec

w2v = Word2vec(3, 150)

w2v.fit('notebooks/cleaned_2.csv')

def write_metadata(w2v, save = 'meta'):
    words = w2v.word_to_id.keys()
    ENCname = save+'.tsv'
    with open(ENCname, "w") as f:
        for i in range(2):
            for word in words:
                f.write(f'{word}\n')


write_metadata(w2v)