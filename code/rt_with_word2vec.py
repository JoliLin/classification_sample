import language_model as LM
import nltk
import numpy as np
import training_handler as trainer
import torch
import tokenizer
from tokenizer import FullTokenizer
from sklearn.model_selection import train_test_split

def load_data( lst ):
    x, y = [], []

    for i in lst:
        y_, x_ = i.split(' ', 1)

        x.append(x_)
        y.append(int(y_))

    return x, y

class rt:
    def __init__(self, path = '../dataset/rt-polarity.all'):
        data = open(path, encoding='utf-8', errors='ignore').readlines()

        np.random.seed(1)
        np.random.shuffle(data)

        self.seq_len = 512
        self.wv = LM.pretrained(max_len=self.seq_len, padding_item=[300*[0]])
        self.data = data

    def process(self, x):
        x = map(tokenizer.tokenize, x)
        x = map(tokenizer.remove_stopwords, x)
        x = self.wv.preprocess(x)
        return x

    def simple_data(self):
        x, y = load_data(self.data)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.1)

        train_x = torch.FloatTensor(self.process(train_x))
        dev_x = torch.FloatTensor(self.process(dev_x))
        test_x = torch.FloatTensor(self.process(test_x))

        print(train_x.shape)
        print(dev_x.shape)
        print(test_x.shape)

        return (train_x, torch.LongTensor(train_y)), (dev_x, torch.LongTensor(dev_y)), (test_x, torch.LongTensor(test_y))

if __name__ == '__main__':
    rt = rt()
    training, valid, testing = rt.simple_data()
    train_loader = trainer.load_data(training)
    valid_loader = trainer.load_data(valid)
    test_loader = trainer.load_data(testing)
    
    import model
    trial = 1
    scores = []
    for i in range(trial):
        model_ = model.simple_CNN()
        p = trainer.handler(model_, train_loader, valid_loader, test_loader)
        score = p.fit('model_'+str(i)+'.pt')

        scores.append(score)
        print(scores)
    print(sum(scores)/trial)
