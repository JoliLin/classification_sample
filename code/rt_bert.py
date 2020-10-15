import numpy as np
import torch
from transformers import BertTokenizer#, BertModel
from sklearn.model_selection import train_test_split
from pipeline.BertAdapter import BertModel
from pipeline.training_handler import handler
from pipeline.tokenizer import FullTokenizer

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

        self.weight = 'bert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained(self.weight)

    def process(self, x):
        x_ = [self.tokenizer.encode(i, add_special_tokens=True, max_length=300, padding='max_length', truncation=True) for i in x]

        return x_

    def simple_data(self):
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
    
    #data preprocess
    training, valid, testing = rt.simple_data()
    train_loader = handler.torch_data(training)
    valid_loader = handler.torch_data(valid)
    test_loader = handler.torch_data(testing)
   
    from pipeline import model
    trial = handler.trial
    scores = []
    for i in range(trial):
        #setting
        #model_ = model.BertCls( BertModel, rt.weight, handler.trainable )
        model_ = model.CustomBertCls( BertModel, rt.weight, handler.trainable )

        model_name = 'model_'+str(i)+'.pt'
        model_path = handler.save+model_name
        handler.setting(model_, model_name=model_name)
        #training
        handler.fit(train_loader, valid_loader)
        #predict
        y_pred = handler.predict(model_, test_loader, model_path)
        #evaluation
        score = handler.eval(testing[1], y_pred)
        
        scores.append(score)
        print(scores)
    print(sum(scores)/trial)
