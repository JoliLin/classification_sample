import cupy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
   
class simple_CNN(nn.Module):
    def __init__(self, dim=300, seq_len=300, hidden=64, classes=2):
        super(simple_CNN, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.activation = nn.ReLU()

        self.layer1 = nn.Conv2d(1, hidden, (3, dim), padding=(1, 0))
        self.decoder = nn.Linear(hidden, classes)

        self.dropout = nn.Dropout(p=0.1)
    def main_task(self, h):
        h = self.layer1(h.unsqueeze(1)).squeeze(3)
        h = self.activation(h)
        h = F.avg_pool2d(h, (1, h.size(2))).squeeze(2)
        
        h = self.dropout(h)
        rating = self.decoder(h)
        return rating, h

    def forward(self, data_, mode='train'):
        if mode == 'inference':
            return self.main_task(data_[0])[0]
        else:
            x1 = data_[0]
            y1 = data_[1]

            if len(data_) == 4:
                return
            else:
                y_rating, embed = self.main_task(x1)
                return y_rating, self.loss_func(y_rating, y1)

class EmbCNN(nn.Module):
    def __init__(self, dim=300, seq_len=300, hidden=64, classes=2):
        super(EmbCNN, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.activation = nn.ReLU()

        self.layer1 = nn.Embedding(120000, dim, padding_idx=0)
        self.conv = nn.Conv2d(1, hidden, (3, dim), padding=(1, 0))
        self.decoder = nn.Linear(hidden, classes)
        self.dropout = nn.Dropout(p=0.1)

    def main_task(self, h):
        h = self.layer1(h)
        h = self.dropout(h)
        h = self.conv(h.unsqueeze(1)).squeeze(3)
        h = self.activation(h)
        h = F.avg_pool2d(h, (1, h.size(2))).squeeze(2)
        h = self.dropout(h)
        rating = self.decoder(h)
        return rating, h

    def forward(self, data_, mode='train'):
        if mode == 'inference':
            return self.main_task(data_[0])[0]
        else:
            x1 = data_[0]
            y1 = data_[1]

            if len(data_) == 4:
                return
            else:
                y_rating, embed = self.main_task(x1)
                return y_rating, self.loss_func(y_rating, y1)

class BertCls(nn.Module):
    def __init__(self, bert_model, bert_weight, trainable, classes=2):
        super(BertCls, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.activation = nn.ReLU()
   
        self.pretrained = bert_model.from_pretrained(bert_weight)
       
        if trainable > 0 :
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
            for i in range(1, trainable+1):
                for param in self.pretrained.encoder.layer[-i].parameters():
                    param.requires_grad = True
        
        self.decoder = nn.Linear(768, classes)
        self.dropout = nn.Dropout(p=0.1)

    def main_task(self, h):
        _, h = self.pretrained(h)

        h = self.dropout(h)
        rating = self.decoder(h)
        return rating, h

    def forward(self, data_, mode='train'):
        if mode == 'inference':
            return self.main_task(data_[0])[0]
        else:
            x1 = data_[0]
            y1 = data_[1]

            if len(data_) == 4:
                return
            else:
                y_rating, embed = self.main_task(x1)
                return y_rating, self.loss_func(y_rating, y1)

class DistilBertCls(nn.Module):
    def __init__(self, bert_model, bert_weight, classes=2):
        super(DistilBertCls, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.activation = nn.ReLU()
       
        self.pretrained = bert_model.from_pretrained(bert_weight)

        #DistilBert
        for param in self.pretrained.parameters():
            param.requires_grad = False
        for param in self.pretrained.transformer.layer[-1].parameters():
            param.requires_grad = True

        self.decoder = nn.Linear(768, classes)
        self.dropout = nn.Dropout(p=0.1)

    def main_task(self, h):
        #64x512x768
        #DistilBert
        h = checkpoint(self.pretrained, h)
        h = h[0][:, 0, :]
        
        #_, h = self.pretrained(h)

        rating = self.decoder(h)
        return rating, h

    def forward(self, data_, mode='train'):
        if mode == 'inference':
            return self.main_task(data_[0])[0]
        else:
            x1 = data_[0]
            y1 = data_[1]

            if len(data_) == 4:
                return
            else:
                y_rating, embed = self.main_task(x1)
                return y_rating, self.loss_func(y_rating, y1)

