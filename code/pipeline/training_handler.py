import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import time
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

from .args import hyperparameter
#from replacement_scheduler import ConstantReplacementScheduler
from transformers import get_linear_schedule_with_warmup

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

#training process
class handler(hyperparameter):
    def __init__(self):
        super().__init__()
        self.hist = dict()
   
    def setting(self, model, model_name):
        self.model_ = model
        self.model_name = model_name
    
    def torch_data(self, data, dataset=CustomDataset ):
        return Data.DataLoader( dataset=dataset(*data), batch_size=self.batch_size )

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return max(0.0, float(num_training_steps-current_step)/float(max(1, num_training_steps-num_warmup_steps)))
        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def print_hist(self):
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
    
    def fit( self, train_loader, valid_loader=None, test_loader=None):
        model_ = self.model_ 
        #print(model_)
        total = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        print('# of para: {}'.format(total))	

        es = EarlyStopping(patience=self.patience, verbose=True)

        optimizer = torch.optim.AdamW(model_.parameters(), lr=self.lr*self.accumulative, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(train_loader)//1*self.epoch_size))
        #lr_schedule = 
        model_.to(self.device)

        start = time.time()
        best_model = None
        optimizer.zero_grad()

        for epoch in range(self.epoch_size):
            start = time.time()
		
            model_.train()
            train_loss, valid_loss = 0.0, 0.0
            train_acc = 0.0
            N_train = 0

            for i, data_ in enumerate(train_loader):
                data_ = [_.to(self.device) for _ in data_]
                #optimizer.zero_grad()
                y_pred, loss = model_(data_, mode='train')
                loss = loss / self.accumulative
                train_loss += loss.item()*len(data_[0])
                train_acc += self.accuracy( y_pred, data_[1] ).item()*len(data_[0])
                N_train += len(data_[0])
	
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model_.parameters(), 1)

                ###attack
                '''
                fgm.attack()
                y_pred ,loss_atk = model_(data_, mode='train')
                loss_atk.backward()
                fgm.restore()
                '''
                ###
                ###Gradient Accumulative
                if (i+1) % self.accumulative == 0:
                    torch.nn.utils.clip_grad_norm_(model_.parameters(), 1)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            self.hist['Epoch'] = epoch+1
            self.hist['time'] = time.time()-start
            self.hist['train_loss'] = train_loss/N_train	
            self.hist['train_acc'] = train_acc/N_train

            torch.save(model_.state_dict(), self.save+self.model_name)
			
            if valid_loader != None:
                valid_true, valid_pred, valid_loss = self.test(model_, valid_loader, mode='valid', model_name=self.save+self.model_name)

                es(valid_loss, model_, self.save+self.model_name)
	
                '''###
                self.test(model_, test_loader, mode='valid', model_name=self.save+self.model_name)
		###
                '''

                self.print_hist()

                if es.early_stop:	
                    print('Early stopping')
                    break
            else:
                self.print_hist()

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
        self.hist['val_acc'] = test_acc/N_test
		
        return y_true , y_pred, test_loss/N_test

    def predict( self, model_, test_loader, model_path='checkpoint.pt' ):
        model_.to(self.device)
        model_.load_state_dict(torch.load(model_path))
        model_.eval()

        y_pred = []
        with torch.no_grad():
            for i, data_ in enumerate(test_loader):
                data_ = [_.to(self.device) for _ in data_]
                logit = model_(data_, mode='inference')
                y_pred.extend(logit.detach().cpu())

        return y_pred
            
    def eval(self, y_true, y_pred):
        hit = 0
        for a, b in zip([i.detach().cpu() for i in y_true], y_pred):
            if a == np.argmax(np.array(b)):
                hit+=1
        return hit/len(y_true)

    def accuracy( self, y_pred, y_true ):
        return (np.array(list(map(np.argmax, y_pred.detach().cpu()))) == np.array(y_true.cpu())).sum()/len(y_pred)
