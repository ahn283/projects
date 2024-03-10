# environment.py

class Environment:
    
    '''
    Attributions
    ---------
    - chart_data : stock chart data
    - observation : current observation
    - idx : current index in the chart data
    
    Functions
    ---------
    - reset() : intitialize with idx and observation
    - observe() : move to the next idx and update observation
    - get_price() : get close price of current observation
    '''
    
    PRICE_IDX = 4       # position of close price
    
    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1
        
    def reset(self):
        self.observation = None
        self.idx = -1
        
    def observe(self):
        # move the next index and return observation data
        # if it is the last position, return None
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None
    
    def get_price(self):
        # get close price from the observation data
        # because close price is located in the fifth column, PRICE_INDEX = 4
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None