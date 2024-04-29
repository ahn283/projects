import os
import logging
import abc
import collections
import threading
import time
import json
import numpy as np
from tqdm import tqdm
from environment import Environment
from agent import Agent
from visualizer import Visualizer
import netowrk as net

BASE_DIR = os.path.abspath(os.path.join(os.path.curdir))    # 'C:\\project\\github\\projects\\traders'
LOGGER_NAME = 'trader'
logger = logging.getLogger(LOGGER_NAME)

# common learner class

class ReinforcementLearner:

    ''' 
    Attributes
    --------
    - stock_code
    - stock_data : stock data for plotting chart
    - environment
    - agent
    - training_data : data for training a model
    - value_network : value network for a model if needed
    - policy_network : policy network for a model if needed
    
    Functions
    --------
    - init_value_network() : function for creating value network
    - init_policy_network() : function for creating policy network
    - build_sample() : get samples from environment instances
    - get_batch() : create batch training data
    - update_network() : training vlue network and policy network
    - fit() : request train value network and policy network
    - run() : perform reinfocement learning
    - save_models() : save value network and policy network
    '''
    
    lock = threading.Lock()
    
    def __init__(self, rl_method='dqn', stock_code=None,
                 stock_data=None, training_data=None,
                 min_trading_money=100, max_trading_money=10000,
                 net='cnn', num_steps=5, lr=0.05,
                 gamma=0.9, num_epochs=1000,
                 balance=100000, eps_init=1,
                 value_network=None, policy_network=None,
                 output_path='', reuse_models=True, gen_output=True):
        
        ''' 
        Attributes
        --------
        - rl_method : reinforcement learning method - 'dqn', 'pg', 'ac', 'a2c', 'a3c', 'ppo', 'acer', ...
        - stock_code
        - stock_data
        - training_data
        - min_trading_money
        - max_trading_money
        - net : neural network - 'dnn', 'lstm', 'cnn', 'rnn', 'alex', ...
        - n_steps : sequence length of 2 dimensional networks such as CNN, RNN, LSTM, AlexNet
        - lr : learning rate
        - gamma : discount rate
        - num_epochs : number of traiding epochs (scenarios)
        - balance : initial balance
        - eps_init : initial exploration rate
        - value_network
        - policy_network
        - output_path
        - reuse_model
        - gen_output Whether the results are visualized or not
        '''
        
        # check arguments
        assert min_trading_money > 0
        assert max_trading_money > 0
        assert max_trading_money >= min_trading_money
        assert num_epochs > 0
        assert lr > 0
        
        # set reinforcement learning
        self.rl_method = rl_method
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.eps_init = eps_init
        
        # set environment
        self.stock_code = stock_code
        self.stock_data = stock_data
        self.env = Environment(stock_data)
        
        # set agent
        self.agent = Agent(self.env, balance, min_trading_money, max_trading_money)

        # training data
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        
        # vector size = training data vector size + agent state vector size
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
            
        # set nework
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        
        # visualize module
        self.visualizer = Visualizer()
        
        # memory 
        self.memory_sample = []         # training data sample
        self.memory_action = []         # actions taken
        self.memory_reward = []         # reward obtained
        self.memory_value = []          # predictiojn values for actions
        self.memory_policy = []         # prediction probabilities for actions
        self.memory_pv = []             # portfolio value
        self.memory_num_stocks = []     # number of stocks
        self.memory_exp_idx = []        # exploration index
        
        # exploration epoch info
        self.loss = 0               # loss during epoch
        self.itr_cnt = 0            # number of iterations with profit
        self.exploration_cnt = 0    # number of exploration
        self.batch_size = 0         # number of training
        
        # log output
        self.output_path = output_path
        self.gen_output = gen_output
        
    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = net.DNN(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr,
                # num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )

        elif self.net == 'cnn':
            self.value_network = net.CNN(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
            
        elif self.net == 'lstm':
            self.value_network = net.LSTMN(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
            
        elif self.net == 'alex':
            self.value_network = net.AlexNet(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
            
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)
            
    def init_policy_network(self, shared_network=None, activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = net.DNN(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr,
                shared_network=shared_network,
                activation=activation, loss=loss
            ) 
            
        elif self.net == 'lstm':
            self.policy_network = net.LSTM(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
            
        elif self.net == 'cnn':
            self.policy_network = net.CNN(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
            
        elif self.net == 'alex':
            self.policy_network = net.AlexNet(
                input_dim=self.num_features,
                output_dim=Agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
            
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)
            
    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        
        # reset env
        self.env.reset()
        
        # reset agent
        self.agent.reset()
        
        # reset visualizer
        self.visualizer.clear([0, len(self.stock_data)])
        
        # reset memories
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        
        # reset epoch info
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        
    def build_sample(self):
        # get next index data
        self.env.observe()
        # 44 samples + 3 agent states = 47 features
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None
    
    # abstract method
    @abc.abstractmethod
    def get_batch(self):
        pass
    
    # after generate batch data, call train_on_batch() method to train value network and policy network.
    # value network : DQNLearner, ActorCriticLearner, A2CLearner
    # policy network : PolicyGradientLearner, ActorCrotocLearner, A2CLearner
    # loss value after training is saved as instance. in case of training value and policy probabilities
    def fit(self):
        # generate batch data
        x, y_value, y_policy = self.get_batch()
        # initialize loss
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # update value network
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # update policy network
                loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss
            
    # visualize one complete epoch
    # in case of LSTM, CNN agent, the number of agent's actions, num_stocks, output of value network, output of policy network and portfolio value is less than daily stock data by (num_steps - 1). So we fill (num_steps - 1) meaningless data.
    def visualize(self, epoch_str, num_epochs, eps):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_policy
            
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epochs=num_epochs,
            eps=eps, action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))
        
        
    def run(self, learning=True):
        ''' 
        Arguments
        --------
        - learning : boolean if learning will be done or not
            - True : after training, build value and policy network
            - False : simulation with pretrained model
        '''      
        info = {
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net}'
            f' LR:{self.lr} DF:{self.gamma}'
        }
        with self.lock:
            logger.debug(info)
            
        # start time
        time_start = time.time()
        
        # prepare visualization
        self.visualizer.prepare(self.env.stock_data, info)
        
        # prepare folders for saving results
        if self.gen_output:
            self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
            if not os.path.isdir(self.epoch_summary_dir):
                os.makedirs(self.epoch_summary_dir)
            else:
                for f in os.listdir(self.epoch_summary_dir):
                    os.remove(os.path.join(self.epoch_summary_dir, f))
                    
        # reset info about training
        # save the most highest porfolio value at max_portfolio_value variable
        max_portfolio_value = 0
        # save the count of epochs with profit
        epoch_win_cnt = 0
        
        # iterate epochs
        for epoch in tqdm(range(self.num_epochs)):
            # start time of an epoch
            time_start_epoch = time.time()
            
            # queue for making step samples
            q_sample = collections.deque(maxlen=self.num_steps)
            
            self.reset()
            
            # decaying exploration rate
            if learning:
                eps = self.eps_init * (1 - (epoch / (self.num_epochs - 1)))
            else:
                eps = self.eps_init
                
            for i in tqdm(range(len(self.training_data)), leave=False):
                # create samples
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # save samples until its size becomes as num_steps
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue
                
                # get predicted value of actions
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                # get predicted probabilities of actions
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                    
                # make decistions based on predicted value and probabilities
                # decide actions based on networks or exploration
                # decide actions randomly with epsilon probability or accoring to network output with (1 - epsilon) probability.
                # policy network output is the probabilities that selling or buying increase portfolio value. If output for buying is larger than that for selling, buy the stock. Otherwise sell it.
                # If there is no output of policy network, select the action with the highest output of value network.
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, eps)
                
                # get rewards from action
                reward = self.agent.step(action, confidence)
                
                # save action and the results in the memory
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)
                    
                # update iteration info
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0
                
            # training network after completin an epoch
            if learning:
                self.fit()
                
            # log about an epoch info
            # check the length of epoch number string
            num_epochs_digit = len(str(self.num_epochs))
            # fill '0  as same size as the length of number of epochs
            epoch_str = str(epoch + 1).rjust(num_epochs_digit, '0')
            time_end_epoch = time.time()
            # save time of an epoch
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(
                f'[{self.stock_code}][Epoch {epoch_str}] '
                f'Epsilon:{eps:.4f} # Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}'                
            ) 

            # visualize epoch information
            if self.gen_output:
                if self.num_epochs == 1 or (epoch + 1) % max(int(self.num_epochs / 100), 1) == 0:
                    self.visualize(epoch_str, self.num_epochs, eps)
            
            # update training info
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value
            )
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1
                
        # end time
        time_end = time.time()
        elapse_time = time_end - time_start
        
        # log about training
        with self.lock:
            logger.debug(
                f'[{self.stock_code} Elapsed Time:{elapse_time:.4f}] '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}'
            )
            
    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)
            
    # without training, just predict actions based on samples
    def predict(self):
        # initiate an agent
        self.agent.reset()
        
        # queue for step samples
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # create samples
            next_sample = self.build_sample()
            if next_sample is None:
                break
            
            # save samples as many as num_steps
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue
            
            # prediction based on value and policy network
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample)).tolist()
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample)).tolist()
                
            # decide action based on the network
            result.append((self.env.observe[0], pred_value, pred_policy))
            
        if self.gen_output:
            with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f:
                print(json.dump(result), file=f)
                
        return result