import numpy as np

class Agent:

    ''' 
    Attributes
    --------
    - enviroment : instance of environment
    - initial_balance : initial capital balance
    - min_trading_money : minimum trading money
    - max_trading_money : maximum trading money
    - balance : cash balance
    - num_stocks : obtained stocks
    - portfolio_value : value of portfolios (balance + price * num_stocks)
    - num_buy : number of buying
    - num_sell : number of selling
    - num_hold : number of holding
    - ratio_hold : ratio of holding stocks
    - profitloss : current profit or loss
    - avg_buy_price_ratio : the ratio average price of a stock bought to the current price
    
    Functions
    --------
    - reset() : initialize an agent
    - set_balance() : initialize balance
    - get_states() : get the state of an agent
    - decide_action() : exploration or exploitation behavior according to the policy net
    - validate_action() : validate actions
    - decide_trading_unit() : decide how many stocks are sold or bought
    - act() : act the actions
    '''
    
    # agent stste dimensions
    ## (ratio_hold, profitloss, current price to avg_buy_price ratio)
    STATE_DIM = 3
    
    # trading charge and tax
    TRADING_CHARGE = 0.00015    # trading charge 0.015%
    TRADING_TAX = 0.002          # trading tax = 0.2% 
    
    # action space
    ACTION_BUY = 0      # buy
    ACTION_SELL = 1     # sell
    ACTION_HOLD = 2     # hold
    
    # get probabilities from neural nets
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)      # output number from nueral nets
    
    def __init__(self, env,
                 initial_balance=None, min_trading_money=None, max_trading_money=None):        
        
        # get current price from the environment
        self.env = env
        self.initial_balance = initial_balance
        
        # minumum and maximum trainding price
        self.min_trading_money = min_trading_money
        self.max_trading_money = max_trading_money
        
        # attributes for an agent class
        self.balance = initial_balance
        self.num_stocks = 0
        
        # value of portfolio : balance + num_stocks * {current stock price}
        self.portfolio_value = self.balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        
        # three states of Agent class
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0
        
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0
        self.done = False
        
    def set_initial_balance(self, balance):
        self.initial_balance = balance
        
    def get_states(self):
        # return current profitloss based on close price
        close_price = self.env.get_close_price()
        self.ratio_hold = self.num_stocks * close_price / self.portfolio_value
        self.portfolio_value = self.balance + close_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return (
            self.ratio_hold,
            self.profitloss,        # profitloss = (portfolio_value / initial_balance) - 1
            (self.env.get_close_price() / self.avg_buy_price) if self.avg_buy_price > 0 else 0
        )
        
    def decide_action(self, pred_value, pred_policy, eps):
        # act randomly with epsilon probability, act according to neural network  with (1 - epsilon) probability
        confidence = 0
        
        # if theres is a pred_policy, follow it, otherwise follow a pred_value
        pred = pred_policy
        if pred is None:
            pred = pred_value
            
        # there is no prediction from both pred_policy and pred_value, explore!
        if pred is None:
            eps = 1
        else:
            maxpred = np.max(pred)
            # if values for actions are euqal, explore!
            if (pred == maxpred).all():
                eps = 1
                
            # if the difference between buying and selling prediction policy value is less than 0.05, explore!
            if pred_policy is not None:
                if np.max(pred_policy) - np.min(pred_policy) < 0.05:
                    eps = 1
                    
        # decide whether exploration will be done or not
        if np.random.rand() < eps:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS) 
        else: 
            exploration = False
            action = np.argmax(pred)
            
        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = self.sigmoid(pred[action])
            
        return action, confidence, exploration
    
    def validate_action(self, action):
        # validate if the action is available
        if action == Agent.ACTION_BUY:
            # check if al least one stock can be bought.
            if self.balance < self.env.get_next_open_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # check if there is any sotck that can be sold
            if self.num_stocks <= 0:
                return False
        
        return True
    
    def decide_trading_unit(self, confidence):
        # adjust number of stocks for buying and selling according to confidence level
        if np.isnan(confidence):
            return self.min_trading_money
        
        # set buying price range between self.min_trading_money + added_trading_price [min_trading_money, max_trading_money]
        # in case that confidence > 1 causes the price over max_trading_money, we set min() so that the value cannot have larger value than self.max_trading_money - self.min_trading_money
        # in case that confidence < 0, we set max() so that added_trading_price cannot have negative value.
        added_trading_money = max(min(
            int(confidence * (self.max_trading_money - self.min_trading_money)),
            self.max_trading_money - self.min_trading_money
        ), 0)
        
        trading_price = self.min_trading_money + added_trading_money
        
        return max(int(trading_price / self.env.get_next_open_price()), 1)
    
    def step(self, action, confidence):
        '''
        Arguments
        ---------
        - action : decided action from decide_action() method based on exploration or exploitation (0 or 1)
        - confidence : probability from decide_action() method, the probability from policy network or the softmax probability from value network
        '''
        
        # get the next open price from the environment
        
        open_price = self.env.get_next_open_price()
        
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
        
        # buy
        if action == Agent.ACTION_BUY:
            # decide how many stocks will be bought
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - open_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            
            # if lacks of balance, buy maximum units within the amount of money available
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (open_price * (1 + self.TRADING_CHARGE))),
                    int(self.max_trading_money / open_price)
                )
                
            # total amount of money with trading charge
            invest_amount = open_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks + open_price * trading_unit) / (self.num_stocks + trading_unit)
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1
                
        # sell
        elif action == self.ACTION_SELL:
            # decide how many stocks will be sold
            trading_unit = self.decide_trading_unit(confidence)
            
            # if lacks of stocks, sell maximum units available
            trading_unit = min(trading_unit, self.num_stocks)
            
            # selling amount
            invest_amount = open_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)
            ) * trading_unit
            
            if invest_amount > 0:
                # update average buy price
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks - open_price * trading_unit) / (self.num_stocks - trading_unit) if self.num_stocks > trading_unit else 0
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1
                
        # hold
        elif action == self.ACTION_HOLD:
            self.num_hold += 1
            
        # update portfolio value with close price
        close_price = self.env.get_next_close_price()
        
        self.portfolio_value = self.balance + close_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        
        return self.profitloss    
    
    
    def sigmoid(x):
        x = max(min(x, 10), -10)
        return 1. / (1. + np.exp(-x))