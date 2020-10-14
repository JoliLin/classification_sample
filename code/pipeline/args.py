import argparse
import os 

class hyperparameter:
    def __init__(self):
        self.parser = self.process_command()
        ###model
        self.gpu = self.parser.gpu
        self.device = self.device_setting(self.gpu)
        self.epoch_size = self.parser.epoch
        self.batch_size = self.parser.batch
        self.lr = self.parser.lr
        self.patience = self.parser.patience
        self.seq_len = self.parser.length
        self.accumulative = self.parser.accumulative
        ###Experiment
        self.trial = self.parser.trial
        self.save = self.create_saving_dir(self.parser.save)

        ###BERT 
        self.trainable = self.parser.trainable

    def process_command(self):
        parser = argparse.ArgumentParser(prog='Training', description='Arguments')
        parser.add_argument('--gpu', '-g', default=0, help='-1=cpu, 0, 1,...= gpt', type=int)
        parser.add_argument('--epoch', '-e', default=300, type=int)
        parser.add_argument('--batch', '-b', default=64, help='batch size', type=int)
        parser.add_argument('--lr', '-lr', default=3e-4, help='learning rate', type=float)
        parser.add_argument('--patience', '-p', default=7, help='patience for learning rate', type=int)	
        parser.add_argument('--length', '-l', default=300, help='sequence length', type=int)

        parser.add_argument('--trial', '-t', default=1, help='times of trial', type=int)	
        parser.add_argument('--save', '-s', default='../__model__/', help='path of saving model')

        parser.add_argument('--trainable', '-trainable', default=9, help='number of BERT trainable layers', type=int)	
        parser.add_argument('--accumulative', '-accumulative', default=2, help='gradient accumulative', type=int)
        return parser.parse_args()

    def device_setting(self, gpu=1):
        return 'cpu' if gpu == -1 else 'cuda:{}'.format(gpu)

    def create_saving_dir(self, save):
        if not save.endswith('/'):
            save = save + '/'
        try:
            os.mkdir(save)
        except FileExistsError:
            print(f'Dir : {save} existed.')
        return save
