import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

import args
import numpy as np
import os 
import time

#data 4 collabrative filtering
class CFDataset(Dataset):
    def __init__(self, u, i, y):
        self.u = Variable(u)
        self.i = Variable(i)
        self.y = Variable(y)
        self.len = len(u)
	
    def __getitem__(self, index):
        return self.u[index], self.i[index], self.y[index]
	
    def __len__(self):
        return self.len

#data 4 multi-task learning
class MultiDataset(Dataset):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = Variable(x1)
        self.y1 = Variable(y1)
        self.x2 = Variable(x2)
        self.y2 = Variable(y2)

        self.len = len(x1)
	
    def __getitem__(self, index):
        return self.x1[index], self.y1[index], self.x2[index], self.y2[index]

    def __len__(self):
        return self.len

#simple data 
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = Variable(x)
        self.y = Variable(y)
        self.len = len(x)
	
    def __getitem__(self, index):
        return self.x[index], self.y[index]
	
    def __len__(self):
        return self.len

def load_data( data, batch_size=64, dataset=CustomDataset ):
    return Data.DataLoader( dataset=dataset(*data), batch_size=batch_size )

#early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, model_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, model_name):
        torch.save(model.state_dict(), model_name)
        self.val_loss_min = val_loss

#training processes
class handler:
    def __init__(self, model_, train_loader, valid_loader, test_loader):
        arg = args.process_command()	    

        self.gpu = arg.gpu
        self.epoch_size = arg.epoch
        self.batch_size = arg.batch
        self.device = self.device_setting( self.gpu )
        self.save = arg.save
        
        self.model_ = model_
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.hist = dict()

        self.create_saving_dir(self.save)

    def create_saving_dir( self, save ):
        try:
            os.mkdir(save)
        except FileExistsError:
            print('Dir : {} existed.'.format(save))

    def device_setting( self, gpu=1 ):
        return 'cpu' if gpu == -1 else 'cuda:{}'.format(gpu)

    def print_hist( self ):
        string = '> '

        for k in self.hist.keys():
            val = self.hist[k]

            if type(val) == float:
                string += '{}:{:.6f}\t'.format(k,val)	
            else:
                string += '{}:{}\t'.format(k,val)
        print(string)	
		
    def updateBN(self, s):
        for m in model_.modules():
            if isinstance(m, nn,LayerNorm):
                m.weight.grad.data.add_(s*torch.sign(m.weight.data))
    
    def train( self, model_, train_loader, valid_loader, model_name='checkpoint.pt' ):
        es = EarlyStopping(patience=10, verbose=True)

        #optimizer = Lamb(model_.parameters())
        optimizer = torch.optim.AdamW(model_.parameters())
        
        #DistilBert
        #optimizer = torch.optim.Adam([{'params':model_.pretrained.transformer.layer[-1].parameters(), 'lr': 5e-5},{'params':model_.decoder.parameters(), 'lr':3e-4}])
        #optimizer = torch.optim.SGD(model_.parameters(), lr=3e-4, momentum=0.9)	
        
        #Bert
        #optimizer = torch.optim.Adam([{'params':model_.pretrained.encoder.layer[-1].parameters(), 'lr': 5e-5},{'params':model_.pretrained.encoder.layer[-2].parameters(), 'lr': 5e-5}, {'params':model_.pretrained.encoder.layer[-3].parameters(), 'lr': 5e-5}, {'params':model_.pretrained.encoder.layer[-4].parameters(), 'lr': 5e-5}, {'params':model_.decoder.parameters(), 'lr':3e-4}])

        model_.to(self.device)

        start = time.time()
        best_model = None
	
        for epoch in range(self.epoch_size):
            start = time.time()
		
            model_.train()
            train_loss, valid_loss = 0.0, 0.0
            train_acc = 0.0
            N_train = 0

            for i, data_ in enumerate(train_loader):
                data_ = [_.to(self.device) for _ in data_]
                optimizer.zero_grad()
                y_pred, loss = model_(data_, mode='train')
				
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 5)
	
                train_loss += loss.item()*len(data_[0])
                train_acc += self.accuracy( y_pred, data_[1] ).item()*len(data_[0])
                optimizer.step()
                N_train += len(data_[0])

            self.hist['Epoch'] = epoch+1
            self.hist['time'] = time.time()-start
            self.hist['train_loss'] = train_loss/N_train	
            #self.hist['train_acc'] = train_acc/N_train

            torch.save(model_.state_dict(), model_name)
			
            if valid_loader != None:
                valid_true, valid_pred, valid_loss = self.test(model_, valid_loader, mode='valid', model_name=model_name)

                es(valid_loss, model_, model_name)
				
            self.print_hist()

            if es.early_stop:	
                print('Early stopping')
                break
				
    def test( self, model_, test_loader, mode='test', model_name='checkpoint.pt' ):
        model_.load_state_dict(torch.load(model_name))
        model_.to(self.device)
        model_.eval()
        test_loss = 0.0
        test_acc = 0.0

        y_pred, y_true = [], []
        N_test = 0
        with torch.no_grad():
            for i, data_ in enumerate(test_loader):
                data_ = [_.to(self.device) for _ in data_]
                logit, loss = model_(data_, mode=mode)

                y_pred.extend(logit)
                y_true.extend(data_[1])
                test_loss += loss.item()*len(data_[0])
                test_acc += self.accuracy( logit, data_[1]).item()*len(data_[0])
                N_test += len(data_[0])

        self.hist['val_loss'] = test_loss/N_test
        #self.hist['val_acc'] = test_acc/N_test
		
        return y_true , y_pred, test_loss/N_test

    def predict( self, model_, test_loader, model_name='checkpoint.pt' ):
        model_.to(self.device)
        model_.load_state_dict(torch.load(model_name))
        model_.eval()

        y_pred = []
        with torch.no_grad():
            for i, data_ in enumerate(test_loader):
                data_ = [_.to(self.device) for _ in data_]
                logit = model_(data_, mode='inference')
                y_pred.extend(logit.detach().cpu())

        return y_pred

    def fit(self, model_name='MLP.pt'):
        #print(model_)
        total = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        print('# of para: {}'.format(total))	

        ### train & test
        self.train( self.model_, self.train_loader, self.valid_loader, self.save+model_name )
        y_true, y_pred, avg_loss = self.test( self.model_, self.test_loader, model_name=self.save+model_name )
        hit = 0
        for a, b in zip([i.detach().cpu() for i in y_true], [i.detach().cpu() for i in y_pred]):
            if a == np.argmax(np.array(b)):
                hit+=1
        return hit/len(y_true)

    def inference(self, model_name='MLP.pt'):
        #print(model_)
        total = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        print('# of para: {}'.format(total))	
        
        predicted = self.predict(self.model_, self.test_loader, model_name)
        print([np.argmax(np.array(i)) for i in predicted])

    def accuracy( self, y_pred, y_true ):
        return (np.array(list(map(np.argmax, y_pred.detach().cpu()))) == np.array(y_true.cpu())).sum()/len(y_pred)
