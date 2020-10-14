import numpy as np
import torch
from pipeline.training_handler import handler
from pipeline.tokenizer import FullTokenizer
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

        self.seq_len = 300
        self.data = data
        self.tokenizer = FullTokenizer('../vocab.txt')

    def process(self, x):
        padding_head = lambda a: ['[CLS]']+a[:(self.seq_len-2)]+['[SEP]'] if len(a) > self.seq_len-2 else ['[CLS]']+a[:(self.seq_len-2)]+['[SEP]']  
        padding_tail = lambda a : a[:self.seq_len] if len(a) > self.seq_len else a+[0]*(self.seq_len-len(a))
        x = map(self.tokenizer.tokenize, x)
        x = map(padding_head, x)
        x = map(self.tokenizer.convert_tokens_to_ids, x)
        x = map(padding_tail, x)

        return list(x)
  
    def simple_data(self):
        wv = []
        x, y = load_data(self.data)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.1)

        train_x = torch.LongTensor(self.process(train_x))
        dev_x = torch.LongTensor(self.process(dev_x))
        test_x = torch.LongTensor(self.process(test_x))

        print(train_x.shape)
        print(dev_x.shape)
        print(test_x.shape)

        return (train_x, torch.LongTensor(train_y)), (dev_x, torch.LongTensor(dev_y)), (test_x, torch.LongTensor(test_y))
    
if __name__ == '__main__':
    rt = rt()
    handler = handler()

    training, valid, testing = rt.simple_data()
    train_loader = handler.torch_data(training)
    valid_loader = handler.torch_data(valid)
    test_loader = handler.torch_data(testing)
   
    from pipeline import model
    #import attention
    trial = handler.trial
    scores = []
    for i in range(trial):
        #model_ = attention.attention(emb=64, n_head=8)
        model_ = model.EmbCNN(dim=768)
        model_name = 'model_'+str(i)+'.pt'
        handler.setting(model_, model_name=model_name)
        handler.fit(train_loader, valid_loader)
        y_pred = handler.predict(model_, test_loader, handler.save+model_name)
        score = handler.eval(testing[1], y_pred)
        
        scores.append(score)
        print(scores)
    print(sum(scores)/trial)
