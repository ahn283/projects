import threading
import abc
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# common network class

class Network:
    
    '''
    Common attributes and methods for neural networks
    
    Attributes
    --------
    - input_dim
    - output_dim
    - lr : learning rate
    - shared_network : head of neural network which is shared with various networks (e.g., A2C)
    - activation : activation layer function ('linear', 'sigmoid', 'tanh', 'softmax')
    - loss : loss function for networks
    - model : final neural network model
    
    Functions
    --------
    - predict() : calculate value or probability of actions
    - train_on_batch() : generate batch data for training
    - save_model()
    - load_model()
    - get_shared_network() : generate network head for the networks
    '''
    
    # threading lock for A3C
    lock = threading.Lock()
    
    def __init__(
        self, input_dim=0, output_dim=0, num_steps=1, lr=0.001,
        net=None, shared_network=None, activation='sigmoid', loss='mse'
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.net = net

        # data shape for various network
        # CNN, LSTM have 3 dimensional shape, so we set input shape as (num_stpes, input_dim)
        # DNN have 2 dimensional shape and we set input shape as (input_dim,)
        inp = None
        if self.num_steps > 1:
            inp = (self.num_steps, input_dim)
        else:
            inp = (self.input_dim,)
            
        # in case that shared network is used,
        self.head = None
        if self.shared_network is None:
            self.head = self.get_network_head(inp, self.output_dim)
        else:
            self.head = self.shared_network
            
        # neual network model
        ## generate network model for head
        self.model = nn.Sequential(self.head)

        # add activation layer
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('activation', nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', nn.Softmax(dim=1))
        self.model.apply(Network.init_weights)
        self.model.to(DEVICE)

        # optimizer
        self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        
        # loss function
        self.criterion = None
        if loss == 'mse':
            self.criterion = nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = nn.BCELoss()
    
    def predict(self, sample):
        # return prediction of buy, sell and hold on given sample
        # value network returns each actions' value on sample and policy network returns each actions' probabilities on sample
        with self.lock:
            # transform evaluation mode: deactivate module used only on training such as Dropout
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(sample).float().to(DEVICE)
                pred = self.model(x).detach().cpu().numpy()
                pred = pred.flatten()
            return pred

    def train_on_batch(self, x, y, a=None, eps=None, K=None):
        if self.num_steps > 1:
            x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        else:
            x = np.array(x).reshape((-1, self.input_dim))
            
        loss = 0.
        
        if self.net == 'ppo':
            with self.lock():
                self.model.train()
                _x = torch.from_numpy(x).float().to(DEVICE)
                _y = torch.from_numpy(y).float().to(DEVICE)
                probs = F.softmax(_y, dim=1)
                for _ in range(K):
                    y_pred = self.model(_x)
                    probs_pred = F.softmax(y_pred, dim=1)
                    rto = torch.exp(torch.log(probs[:, a]) - torch.log(probs_pred[:, a]))
                    rto_adv = rto * _y[:, a]
                    clp_adv = torch.clamp(rto, 1 - eps, 1 + eps) * _y[:, a]
                    _loss = -torch.min(rto_adv, clp_adv).mean()
                    self.optimizer.zero_grad()
                    _loss.backward()
                    self.optimizer.step()
                    loss += _loss.item()
        
        else:
            with self.lock:
                self.model.train()
                _x = torch.from_numpy(x).float().to(DEVICE)
                _y = torch.from_numpy(y).float().to(DEVICE)
                y_pred = self.model(_x)
                _loss = self.criterion(y_pred, _y)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                loss += _loss.item()
            return loss
        
    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        if net == 'dnn':
            return DNN.get_network_head((input_dim,), output_dim)
        elif net == 'lstm':
            return LSTM.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'cnn':
            return CNN.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'alex':
            return AlexNet.get_network_head((num_steps, input_dim), output_dim)
        
    @abc.abstractmethod
    def get_network_head(inp, output_dim):
        pass
    
    @staticmethod
    def init_weights(m):
        # initialize weights as weighted normal distribution
        if isinstance(m, nn.Linear) or isinstance(m, torch.nn.Conv1d):
            nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    nn.init.normal_(weight, std=0.01)
                    
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)
    
    def load_model(self, model_path):
        if model_path is not None:
            self.model = torch.load(model_path)
            
# DNN network class

class DNN(Network):
    @staticmethod
    def get_network_head(inp, output_dim):
        return nn.Sequential(
            nn.BatchNorm1d(inp[0]),         # input.shape = (input_dim, )
            nn.Linear(inp[0], 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_dim),
        )
        
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)
    
class LSTM(Network):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_network_head(inp, output_dim):
        return nn.Sequential(
            nn.BatchNorm1d(inp[0]),
            LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_dim)
        )
        
    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
        return super().predict(sample)
        
class LSTMModule(nn.LSTM):
    def __init__(self, *args, use_last_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_last_only = use_last_only
        
    def forward(self, x):
        output, (h_n, _) = super().forward(x)
        if self.use_last_only:
            return h_n[-1]
        return output
    
class CNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2
        return nn.Sequential(
            nn.BatchNorm1d(inp[0]),
            nn.Conv1d(inp[0], 1, kernel_size),
            nn.BatchNorm1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(inp[1] - (kernel_size - 1), 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_dim),
        )
        
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)
    
class AlexNet(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2,
        stride = 2,
        padding = 1,
        return nn.Sequential(
            nn.BatchNorm1d(inp[0]),
            nn.Conv1d(inp[0], 96, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 256, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding),  
            nn.Conv1d(256, 384, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, kernel_size=kernel_size, padding=padding),
            nn.Conv1d(384, 256, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Flatten(),
            
            # classifier
            # nn.Flatten(6),
            nn.Linear(1536, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_dim),       
        )
        
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)
        

class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        
    def forward(self, x):
        # Do print / debug stuff
        print(x.size())
        return x