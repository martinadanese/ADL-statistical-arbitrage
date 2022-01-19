import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


#   --------------------------------------------------------- 
class Preprocessor:
#   --------------------------------------------------------- 
    """
    Class for handling data preprocessing. 
    
    Members:
    
     - df_open  : dataframe with open positions for the tarining period
     - df_close : dataframe with close positions for the tarining period

    Functions:

     - create_label  :  computes the label for each stock in each day based on their intraday return
     - create_features_for_single_ticker  :  computer the training and test set for a single ticker
     - normalize_data : normalize data with sklearn RobustScaler

    """
#   --------------------------------------------------------- 
   

    def __init__(self, df_open, df_close):

        self.df_open = df_open 
        self.df_close = df_close
        



    #   ----------------------------------------------------- 
    def create_label(self, quantiles=[0.5,0.5]):
    #   ----------------------------------------------------- 
    
        """
        Compute the label for each stock in each day:
        if the current intraday return of a stock is higher [lower] than the cross-sectional median
        of all the stocks of that day the label is 1 [0]. 
        The current intraday return: ir_t =  (close_position/open_position) - 1
            
        ------
        input  : None
        output : label dataframe

        """
    #   ----------------------------------------------------- 
        
        if not np.all(self.df_close.iloc[:,0]==self.df_open.iloc[:,0]):
            print('Date Index issue')
            return
        
        #create quantiles used in data labelling
        quantiles = [0.]+list(np.cumsum(quantiles))
        # use quantiles (pd.qcut) to get if stock is above or below median
        self.label = (self.df_close.iloc[:,1:].div(self.df_open.iloc[:,1:].values)-1).apply(
                lambda x: pd.qcut(x.rank(method='first'), quantiles, labels=False), axis=1)
        # remove first element beacuse will be discarded in fearure_creation
        self.label = self.label[1:]
        return self.label






    #   ----------------------------------------------------- 
    def create_features_for_single_ticker(self, ticker, test_year, feature_creation_space_size=240):
    #   ----------------------------------------------------- 
    
        """
        Compute the time dependent features for each ticker. For each time value in the range of the feature part
        varying t, the current timestamp, compute for each stock s:
         - The intraday return:                              IR^(s)_(t,k)  =  cp^(s)_(t-k) / op^(s)_(t-k)   - 1;
         - The next day return:                              NR^(s)_(t,k)  =  op^(s)_(t-k) / cp^(s)_(t-k-1) - 1;
         - The returns w.r.t. the previous closing price :   CPR^(s)_(t,k) =  cp^(s)_(t-k) / cp^(s)_(t-k-1) - 1;
        these are assembled in train and test set

        ------
        input  : ticker [string], year used for the test, feature_creation_space_size [integer, default=240]
        output : training and test data for ticker

        """
    #   ----------------------------------------------------- 

        ticker_data = pd.DataFrame([])
        ticker_data['Date'] = list(self.df_close['Date'])
        ticker_data['Name'] = [ticker]*len(ticker_data)
        intraday_return = self.df_close[ticker]/self.df_open[ticker]-1

        #intraday return
        for k in range(feature_creation_space_size)[::-1]:
            ticker_data['IntradayReturn_Feature'+str(k)] = intraday_return.shift(k)

        #next day return
        nextday_return = (np.array(self.df_open[ticker][1:])/np.array(self.df_close[ticker][:-1])-1)
        nextday_return = pd.Series(list(nextday_return)+[np.nan])     
        for k in range(feature_creation_space_size)[::-1]:
            ticker_data['NextdayReturn_Feature'+str(k)] = nextday_return.shift(k)
        
        # return w.r.t. closing price
        close_change = self.df_close[ticker].pct_change()
        for k in range(feature_creation_space_size)[::-1]:
            ticker_data['ClosePercentageReturn_Feature'+str(k)] = close_change.shift(k)
        
        ticker_data['IntradayReturnFuture'] = intraday_return.shift(-1)    
        ticker_data['label'] = list(self.label[ticker])+[np.nan] 
        ticker_data['Month'] = list(self.df_close['Date'].str[:-3])
        ticker_data = ticker_data.dropna()

        trade_year = ticker_data['Month'].str[:4]
        ticker_data = ticker_data.drop(columns=['Month'])
        ticker_train_data = ticker_data[trade_year<str(test_year)]
        ticker_test_data = ticker_data[trade_year==str(test_year)]
        
        return np.array(ticker_train_data),np.array(ticker_test_data)





    #   ----------------------------------------------------- 
    def normalize_data(self, train_data, test_data):
    #   ----------------------------------------------------- 
    
        """
        Normalize data with sklearn RobustScaler.

        ------
        input  : unnormalized train and test set
        output : normalized train and test set

        """
    #   ----------------------------------------------------- 

        scaler = RobustScaler()
        scaler.fit(train_data[:,2:-2])
        train_data[:,2:-2] = scaler.transform(train_data[:,2:-2])
        test_data[:,2:-2] = scaler.transform(test_data[:,2:-2]) 
        return train_data, test_data

