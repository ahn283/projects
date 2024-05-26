import numpy as np
import matplotlib.pyplot as plt
import datetime
import threading
from agent import Agent


from mplfinance.original_flavor import candlestick_ohlc

lock = threading.Lock()

class Visualizer:
    '''
    Attributes
    ----------
    - fig : matplotlib Figure instance plays like a canvas
    - plot() : print charts except daily stock data chart
    - save() : save Figure as an image file
    - clear() : initialize all chart but daily stock data chart
    Returns
    ---------
    - Figure title : parameter, epsilon
    - Axes 1 : daily price chart
    - Axes 2 : number of stocks and agent action chart
    - Axes 3 : value network chart
    - Axes 4 : policy network and epsilon chart
    - Axes 5 : Portfolio value and learning point chart
    '''
    
    COLORS = ['r', 'b', 'g']
    
    def __init__(self):
        self.canvas = None
        self.fig = None
        self.axes = None
        self.title = ''
        self.x = []
        self.xticks = []
        self.xlabels = []
        
    def prepare(self, stock_data, title):
        self.title = title
        
        # shares x-axis among all charts
        # self.x =np.arange(stock_data['date])
        # self.x_label = [datetime.strptime(date, '%Y%m%d').date() for date in stock_data['date']]
        with lock:
            # prepare for printing five charts
            self.fig, self.axes = plt.subplots(
                nrows=5, ncols=1, facecolor='w', sharex=True
            )
            for ax in self.axes:
                # deactivate scientific marks
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                # change y-axis to the right
                ax.yaxis.tick_right()
                
            # chart 1. plot daily stock data
            self.axes[0].set_ylabel('Env.')
            x = np.arange(len(stock_data))
            # make two dimensional array with open, high, low and close order
            ohlc = np.hstack((
                x.reshape(-1, 1), np.array(stock_data)[:, 1:5]
            ))
            # red for positive, blue for negative
            candlestick_ohlc(self.axes[0], ohlc, colorup='r', colordown='b')
            
            # visualize volume
            ax = self.axes[0].twinx()
            volume = np.array(stock_data)[:, 5].tolist()
            ax.bar(x, volume, color='b', alpha=0.3)
            
            # set x-axis
            self.x = np.arange(len(stock_data['date']))
            self.xticks = stock_data.index[[0, -1]]
            self.xlabels = stock_data.iloc[[0, -1]]['date']
            
    def plot(self, epoch_str=None, num_epochs=None, eps=None,
             action_list=None, actions=None, num_stocks=None,
             outvals_value=[], outvals_policy=[], exps=None,
             initial_balance=None, pvs=None):
        ''' 
        Attributes
        ---------
        - epoch_str : epoch for Figure title
        - num_epochs : number of total epochs
        - epsilon : exploration rate
        - action_list : total action list of an agent
        - num_stocks : number of stocks 
        - outvals_value : output array of value network
        - outvals_policy : output array of policy network
        - exps : array whether exploration is true or not
        - initial_balance
        - pvs : array of portfolio values
        '''
        
        with lock:
            # action, num_stocks, outvals_value, outvals_policy, pvs has same size
            # create an array with same size as actions and use same x-axis
            actions = np.array(actions)         # action array of an agent
            
            # turn value network output into an array
            outvals_value = np.array(outvals_value)
            
            # turn policy network output into an array
            outvals_policy = np.array(outvals_policy)
            
            # turn initial balance into an array
            pvs_base = np.zeros(len(actions)) + initial_balance     # array([initial_balance, initial_balance, initial_balance, ...])
            
            # chart 2. plot agent states (action, num_stocks)
            for action, color in zip(action_list, self.COLORS):
                for i in self.x[actions == action]:
                    # express actions as background color : red for buying, blue for selling
                    self.axes[1].axvline(i, color=color, alpha=0.1)
            self.axes[1].plot(self.x, num_stocks, '-k')     # plot number of stocks
            
            # chart 3. plot value network (prediction value for action)
            if (len(outvals_value)) > 0:
                max_actions = np.argmax(outvals_value, axis=1)
                for action, color in zip(action_list, self.COLORS):
                    # plot background
                    for idx in self.x:
                        if max_actions[idx] == action:
                            self.axes[2].axvline(idx, color=color, alpha=0.1)
                    
                    # plot value network
                    ## red for buying, blue for selling, green for holding
                    ## if there are no predicions for action, plot green chart
                    self.axes[2].plot(self.x, outvals_value[:, action], color=color, linestyle='-')
                    
            # chart 4. plot policy network
            # plot exploration as yellow background
            for exp_idx in exps:
                self.axes[3].axvline(exp_idx, color='y')
            
            # plot action as background color
            _outvals = outvals_policy if len(outvals_policy) > 0 else outvals_value
            for idx, outval in zip(self.x, _outvals):
                color = 'white'
                if np.isnan(outval.max()):
                    continue
                # with no exploration area, red for buying, blie for selling
                if outval.argmax() == Agent.ACTION_BUY:
                    color = self.COLORS[0]      # red for buying
                elif outval.argmax() == Agent.ACTION_SELL:
                    color = self.COLORS[1]      # blue for selling
                elif outval.argmax() == Agent.ACTION_HOLD:
                    color = self.COLORS[2]      # green for holding
                self.axes[3].axvline(idx, color=color, alpha=0.1)
                
            # plot policy network
            # red for buying policy network output, blue for selling policy network output
            # when red line is above blue line, buy stocks, otherwise sell stocks
            if len(outvals_policy) > 0:
                for action, color in zip(action_list, self.COLORS):
                    self.axes[3].plot(
                        self.x, outvals_policy[:, action],
                        color=color, linestyle='-'
                    )
                    
            # chart 5. portfolio value
            # horizontal line for initial balance
            self.axes[4].axhline(
                initial_balance, linestyle='-', color='gray'
            )
            
            self.axes[4].fill_between(
                self.x, pvs, pvs_base,
                where=pvs > pvs_base, facecolor='r', alpha=0.1
            )
            self.axes[4].plot(self.x, pvs, '-k')
            self.axes[4].xaxis.set_ticks(self.xticks)
            self.axes[4].xaxis.set_ticklabels(self.xlabels)
            
            # epoch and exploration rate
            self.fig.suptitle(f'{self.title}\nEPOCH:{epoch_str}/{num_epochs} EPSILON:{eps:.2f}')
            # adjust canvas layout
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.85)
            
    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            # initialize charts except non changeable value (stock data)
            for ax in _axes[1:]:
                ax.cla()            # initialize chart
                ax.relim()          # initialize limit
                ax.autoscale()      # reset scale
            
            # reset y-axis label
            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_ylabel('Value')
            self.axes[3].set_ylabel('Policy')
            self.axes[4].set_ylabel('Portfolio')
            for ax in _axes:
                ax.set_xlim(xlim)       # set limit in x-axis
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                # set equal width horizontally
                ax.ticklabel_format(useOffset=False)
                
    def save(self, path):
        with lock:
            self.fig.savefig(path)