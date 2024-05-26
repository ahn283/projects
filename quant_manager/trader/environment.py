import numpy as np

class Environment:

    ''' 
    Attribute
    ---------
    - stock_data : stock price data such as 'open', 'close', 'high', 'low', 'volume'
    - state : current state
    - idx : current postion of stock data
    
    
    Functions
    --------
    - reset() : initialize idx and state
    - observe() : move idx into next postion and get a new state
    - get_close_price() : get close price of current state
    - get_next_close_price() : get close price of next index state
    - get_open_price() : get open price of current state
    - get_next_open_price() : get open price of next indext state
    - get_state() : get current state
    '''
    
    def __init__(self, stock_data=None):
        self.close_price_idx = 4    # index postion of close price
        self.open_price_idx = 1     # index position of open price
        self.stock_data = stock_data
        self.state = None
        self.idx = -1
        
    def reset(self):
        self.state = None
        self.idx = -1
        

    def observe(self):
        # move to next day and get price data
        # if there is no more idx, return None
        if len(self.stock_data) > self.idx + 1:
            self.idx += 1
            self.state = self.stock_data.iloc[self.idx]
            return self.state
        return None
    
    def get_close_price(self):
        # return close price
        if self.state is not None:
            return self.state[self.close_price_idx]
        return None
    
    def get_next_close_price(self):
        # return tomorrow close price
        # if self.idx is the last index, return current close price
        try:
            return self.stock_data.iloc[self.idx + 1, self.close_price_idx]
        except:
            return self.stock_data.iloc[self.idx, self.close_price_idx]
        
    def get_open_price(self):
        # return open price
        if self.state is not None:
            return self.state[self.open_price_idx]
        
    def get_next_open_price(self):
        # return tomorrow open price
        # if self.idx is the last index, return current open price
    
        try:
            return self.stock_data.iloc[self.idx + 1, self.open_price_idx]
        except:
            return self.stock_data.iloc[self.idx, self.open_price_idx] 
        
    def get_state(self):
        # return current state
        if self.state is not None:
            return self.state
        return None