import numpy as np
class pretrained:
    def __init__(self, max_len=300, padding_item=[300*[0]], lang='en'):
        if lang == 'en':
            from gensim.models import KeyedVectors
            self.model = KeyedVectors.load_word2vec_format('../model.bin', binary=True)
        else:
            from gensim.models import KeyedVectors, Word2Vec
            self.model = Word2Vec.load('../zh/zh.bin')

        self.max_len = max_len
        self.padding_item = padding_item

    def load_LM(self):
        return self.LM

    def word2vec(self, w):
        try:
            return np.array(self.model[w])
        except KeyError:
            return np.array([0])
    
    def preprocess(self, corpus):
        zero = 0
        non_zero = 0

        embeddings = [map(self.word2vec, c) for c in corpus]

        for i in range(len(embeddings)):
            _ = list(embeddings[i])

            length = len(_)
            for j in range(length):
                if len(_[j]) < self.max_len:
                    lower = j-1 if j-1 > -1 else 0
                    higher = j+1 if j+1 < length else length-1

                    if len(_[lower]) < 300 and len(_[higher]) < 300:
                        _[j] = np.array(300*[0])
                        zero += 1
                    else:
                        _[j] = (_[lower]+_[higher])/2
                        non_zero += 1
            if length < self.max_len:
                embeddings[i] = _ + (self.max_len-length)*self.padding_item
            else:
                embeddings[i] = _[:self.max_len]

        #print(zero, non_zero)

        return embeddings
