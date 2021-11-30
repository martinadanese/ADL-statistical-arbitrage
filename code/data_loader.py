import numpy as np
import pandas as pd
#!pip install pandas-datareader --upgrade
import pandas_datareader.data as reader
from pandas_datareader._utils import RemoteDataError
import datetime as dt
from dateutil.relativedelta import relativedelta
from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
from config import *
import quandl
import os

class DataLoader:
    """
    Class for handling data loading
    """
    
    def __init__(self, start=None, end=None, reload_sp500=True, overwrite=False, number_of_data=None):

        #define start and end dates
        if start is None:
            self.start = dt.datetime(1990, 1, 1)
        else:
            self.start = start
        
        if end is None:
            self.end = dt.datetime(2015, 1, 1)
        else:
            self.end = end
        
        #read quandl key
        self.quandl_key = quandl_key 
        
        #get sp500 self.tickers
        self.reload_sp500 = reload_sp500
        self.overwrite = overwrite
        if self.reload_sp500:
            self.tickers = self.get_sp500_tickers()
            self.data_folder = '../data/'
        else:
            self.data_folder = '../data/15years_data/'
            self.tickers = [i[0:-4] for i in os.listdir(self.data_folder)]
        
        #load only selected data
        if number_of_data is not None:
            self.tickers = self.tickers[0:number_of_data]
         
        
    def get_sp500_tickers(self):
        """
        input: None
        output: list of tickers in sp500
        """
        
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol']
    

    #def create_date_range(self):
    #    """
    #    """
    #    return rrule(DAILY, dtstart=self.start, until=self.end, byweekday=(MO,TU,WE,TH,FR))


    def retrieve_history(self, ticker, api):
        """
        helper function for get_data. Retrieves history of single stock
        ---
        input: ticker name, desired API
        output: dataframe with history
        """

        if api == 'quandl':
            tmp_df = reader.DataReader("WIKI/"+ticker, 'quandl', self.start, self.end, api_key=self.quandl_key)
        else:
            tmp_df = reader.DataReader(ticker, api, self.start, self.end)
        
        tmp_df.reset_index(inplace=True)
        tmp_df.set_index("Date", inplace=True)
        
        return tmp_df


    def get_data(self, verbose=False):
        """
        Retrieves history of all the selected tickers:
         * using and API (quandl or yahoo finance)
         * from csv
        ---
        input: verbose - boolean.
        output: 2 dataframes, one for open and one for close values.
                columns: tickers
                index: date
        """

        tmp_open = []
        tmp_close = []
        dates = []
        #to track unretrievable data
        found = []
        missed = []
        for ticker in self.tickers:
            
            if verbose:
                print("loading "+ticker)

            if self.reload_sp500:
                # get from api
                
                try: 
                    #default api: quandl
                    api = 'quandl'
                    tmp_df = self.retrieve_history(ticker, api)
                    
                    if self.overwrite is False and os.path.exists(self.data_folder+'{}.csv'.format(ticker)):
                        print('file {}.csv already present'.format(ticker))
                    else:
                        tmp_df.to_csv(self.data_folder+'{}.csv'.format(ticker))
                    
                    # check if dates are fine or need to reload data
                    if len(dates) < len(tmp_df.index):
                        dates = tmp_df.index
                    tmp_open.append(tmp_df['Open'].to_list())
                    tmp_close.append(tmp_df['Close'])
                    found.append(ticker)

                except Exception:
                    try:
                        #second attempt with yahoo finance
                        api = 'yahoo'
                        tmp_df = self.retrieve_history(ticker, api)
                        
                        if self.overwrite is False and os.path.exists(self.data_folder+'{}.csv'.format(ticker)):
                            print('file {}.csv already present'.format(ticker))
                        else:
                            tmp_df.to_csv(self.data_folder+'{}.csv'.format(ticker))
                        if len(dates) < len(tmp_df.index):
                            dates = tmp_df.index
                        tmp_open.append(tmp_df['Open'].to_list())
                        tmp_close.append(tmp_df['Close'])
                        found.append(ticker)
                    except Exception:
                        print("No information for ticker '%s'"%ticker)
                        missed.append(ticker)
                        continue
            else:
                #read from csv
                try:
                    tmp_df = pd.read_csv(self.data_folder+'{}.csv'.format(ticker))
                    tmp_df.index = pd.to_datetime(tmp_df['Date'])
                    mask = (tmp_df.index>self.start) & (tmp_df.index<=self.end)
                    tmp_open.append(tmp_df.loc[mask]['Open'].to_list())
                    tmp_close.append(tmp_df.loc[mask]['Close'].to_list()) 
                    found.append(ticker) 
                except Exception:
                    print("No information for ticker '%s'" % ticker)
                    missed.append(ticker)
                    continue
        
        #write output dataframes
        df_open = pd.DataFrame(tmp_open)
        df_close = pd.DataFrame(tmp_close)
        df_open = df_open.transpose()
        df_close = df_close.transpose()
        df_open.columns = found
        df_close.columns = found
        if self.reload_sp500:
            df_open.set_index(dates, inplace = True)
            df_close.set_index(dates, inplace = True)
        else:
            df_open.set_index(tmp_df['Date'], inplace = True)
            df_close.set_index(tmp_df['Date'], inplace = True)
        
        
        return df_open, df_close
        
