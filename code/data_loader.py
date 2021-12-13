import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import pandas as pd
#!pip install pandas-datareader --upgrade
import pandas_datareader.data as reader
import pandas_datareader as pdreader
from pandas_datareader._utils import RemoteDataError
import datetime as dt
from dateutil.relativedelta import relativedelta
from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
from true_config import *
import quandl
import os

class DataLoader:
    """
    Class for handling data loading
    """
    
    def __init__(self, start=None, end=None, reload_sp500=True, overwrite=False): #number_of_data=None):

        #define start and end dates
        if start is None:
            self.start = dt.datetime(1990, 1, 1)
        else:
            self.start = start
        
        if end is None:
            self.end = dt.datetime(2020, 1, 1)
        else:
            self.end = end
        
        #read quandl key
        self.quandl_key = quandl_key 
        self.alpha_key = alpha_key 
        
        #get sp500 self.tickers
        self.reload_sp500 = reload_sp500
        self.overwrite = overwrite

        #if self.reload_sp500:
        self.data_folder = '../data/'
        #the ones present
        self.tickers = self.get_sp500_tickers()
        
        # remove double classe of the companies entering twice in sp500 (were not considered in the original paper):
        for t in ('GOOGL','DISCK', 'NWS', 'FOX', 'CMCSK', 'UA'):
            if t in self.tickers : self.tickers.remove(t)
        
        #all past and present stock with added and removed date
        self.tickers_history = pd.DataFrame.from_dict({t : [[self.start], [self.end]] for t in self.tickers},
                                 orient='index', columns=['Added_date', 'Removed_date'])
        
    

    #   ----------------------------------------------------- 
    def get_sp500_tickers(self):
    #   ----------------------------------------------------- 
        

        """
        get history of the current sp500 stocks
        
        ------
        input: None
        output: list of tickers in sp500
        """
    #   ----------------------------------------------------- 
        
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol']
    
        def to_datetime(date):
            return dt.datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d')

    

    #   ----------------------------------------------------- 
    def old_tickers_converter(self):
    #   ----------------------------------------------------- 
        
        """
        Convert old ticker names into current one according to Nasdaq data.
        Uses:
         * csv nasdaq_stock_screener.csv with info of 8000 stocks 
               downloadable at https://www.nasdaq.com/market-activity/stocks/screener
         * wikipedia sp500 history to get list of all mentioned tickers
               downloadable at https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
        
        ------
        input: None
        output: dictionary with conversion
               
              {'recorder_ticker' : 'real_NASDAQ_ticker'} for all recorded_tickers
               
        """
    #   ----------------------------------------------------- 
     

        # load wikipedia and nasdaq data 
        df_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        recorded_tickers = pd.concat([df_list[1].Added, df_list[1].Removed], ignore_index=True)
        screener = pd.read_csv('nasdaq_stock_screener.csv')
        
        dict_conversion = {}
        for ticker in recorded_tickers.Security.dropna():
            
            recorded_ticker = recorded_tickers.loc[recorded_tickers.Security==ticker, 'Ticker'].values[0]
            #if in screener
            if any(screener['Name'].str.contains(str(ticker), case=False)):
                # get real nasdaq ticker (more complete -> substring) and store it
                real_ticker = screener.loc[screener['Name'].str.contains(str(ticker), case=False), 'Symbol'].values[0].split('^')[0]
                dict_conversion[recorded_ticker] = real_ticker
            else:
                # leave unchanged
                dict_conversion[real_ticker] = real_ticker

        return dict_conversion
    
    

    #   ----------------------------------------------------- 
    def get_sp500_changes(self):
    #   ----------------------------------------------------- 
        
        """
        Get historical changes of last 20 years of sp500 and stores info into a dataframe
        Uses:
         * old_tickers_converter()
         * wikipedia sp500 history to get list of all sp500 historical constituents of last 20 years
               downloadable at https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
        
        ------
        input: None
        output: dataframe with historical changes. Columns:
           -  Added   : ticker that was added in sp500
           -  Removed : ticker that was removed from sp500
           -  Date    : date of the change
               
        """
    #   ----------------------------------------------------- 

        #get data
        df_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df_changes = pd.DataFrame({'Date' : df_list[1].Date.Date, 'Added' : df_list[1].Added.Ticker, 'Removed' : df_list[1].Removed.Ticker})
        df_changes.Date = pd.to_datetime(df_changes.Date, format='%B %d, %Y')
        
        # remove double classe of the companies entering twice in sp500 (were not considered in the original paper):
        for t in ('GOOGL','DISCK', 'NWS', 'FOX', 'CMCSK', 'UA'):
            df_changes.drop(index=df_changes[(df_changes.Added==t) | (df_changes.Removed==t)].index, inplace=True)
        
        #handle cases in which remove & additions were done in two separate days
        df_changes.fillna(0, inplace=True)
        missing_added = df_changes[df_changes.Added==0].copy(deep=True)
        df_changes.loc[(df_changes.Removed==0), 'Removed'] = missing_added['Removed'].values
        df_changes = df_changes.loc[(df_changes.Added != 0)]
         
        #save to file
        df_changes.to_csv("sp500_changes.csv")
        return df_changes
    

    #   ----------------------------------------------------- 
    def get_tickers_history(self):
    #   ----------------------------------------------------- 
        return self.tickers_history



    #   ----------------------------------------------------- 
    def retrieve_sp500_history(self):
    #   ----------------------------------------------------- 
        
        """
        Creates a history of all historical and current sp500 components identifying
        dates and previous/successive ticker.
        A class dictionary (self.dict_history = {t : [[t],[]] for t in self.tickers}) with 500 keys and a list of two lists as values is created:
           
           - key: i-th ticker
           - value: list of two lists:
               [
                 - I inner list  : [ticker_i, preceiding to ticker_i, preceiding to the preceiding to ticker_i, ...], 
                 - II inner list : [end date of study persiod, date of ticker_i and preceding of ticker_i change, 
                                  date of preceding of ticker_i and preceding of the preceding of ticker_i change, ...]
               ]

        Uses:
         * get_sp500_changes()
         * tickers_rename.csv - csv containing name changes not handled in wikipedia. Manually created with wikipedia and 
           nasdaq publically available infos.
        
        ------
        input: None
        output: None
               
        """
    #   ----------------------------------------------------- 
        # get sp500 changes
        df_changes = self.get_sp500_changes()
        # get name changes not handled in wikipedia history
        df_tickers_rename = pd.read_csv('tickers_rename.csv', index_col=None, header=None, names=['Old_name', 'New_name', 'Date'])
        df_tickers_rename.Date = pd.to_datetime(df_tickers_rename.Date)
        
        #map old ticker names to current ticker names
        dict_conversion = self.old_tickers_converter()
        df_changes['Added'] = df_changes['Added'].map(dict_conversion).fillna(df_changes['Added'])
        df_changes['Removed'] = df_changes['Removed'].map(dict_conversion).fillna(df_changes['Removed'])
        
        self.dict_history = {t : [[t],[]] for t in self.tickers}
        for t in self.tickers:
            
            ticker = t
            previous_date = self.end
            while any(df_changes.Added==ticker):
                # take predecessor and date of ticker addition. If more than one, take most recent (to handel being added and removed more than once)
                predecessor = df_changes.loc[(df_changes.Date == df_changes.loc[df_changes.Added==ticker, 'Date'].max()) & (df_changes.Added==ticker), 'Removed'].values[0]
                date = df_changes.loc[(df_changes.Date == df_changes.loc[df_changes.Added==ticker, 'Date'].max()) & (df_changes.Added==ticker), 'Date'].values[0]
                #remove column from df_changes so that the max date can always be taken
                df_changes = df_changes.drop(df_changes[(df_changes.Date == df_changes.loc[df_changes.Added==ticker, 'Date'].max()) & (df_changes.Added==ticker)].index)
                if pd.to_datetime(date) > pd.to_datetime(previous_date):
                    #some missing record, avoid inconsistencies and loop again
                    continue
                self.dict_history[t][0].append(predecessor)
                self.dict_history[t][1].append(date)
                ticker = predecessor
                previous_date = date

            self.dict_history[t][1].append(self.start) 


        

    #   ----------------------------------------------------- 
    def retrieve_ticker_history(self, ticker, start, end, api):
    #   ----------------------------------------------------- 
        
        """
        helper function that retrieves history of single stock
        
        ------
        input: ticker name, desired API
        output: dataframe with history
        """
    #   ----------------------------------------------------- 
        
        if api == 'quandl':
            tmp_df = reader.DataReader("WIKI/"+ticker, 'quandl', start, end, api_key=self.quandl_key)
            tmp_df.reset_index(inplace=True)
            tmp_df.set_index("Date", inplace=True)
            #quandl writes opposite to other apis
            tmp_df = tmp_df.reindex(index=tmp_df.index[::-1])
        
        elif api == 'alphavantage':
            #alpha is the best but has data only after 2000
            if start > pd.to_datetime('01/01/2000') and end > pd.to_datetime('01/01/2000'):
                ts = pdreader.av.time_series.AVTimeSeriesReader(ticker, start=start, end=end, api_key = self.alpha_key)
                tmp_df = ts.read()
                tmp_df.columns = map(str.title, tmp_df.columns)
                tmp_df.index = pd.to_datetime(tmp_df.index, format='%Y-%m-%d')
            
            elif start < pd.to_datetime('01/01/2000') and end > pd.to_datetime('01/01/2000'):
                ts = pdreader.av.time_series.AVTimeSeriesReader(ticker, start='01/01/2000', end=end, api_key = self.alpha_key)
                tmp_df = ts.read()
                tmp_df.columns = map(str.title, tmp_df.columns)
                tmp_df.index = pd.to_datetime(tmp_df.index, format='%Y-%m-%d')
                tmp_df2 = retrieve_ticker_history(self, ticker, start, '01/01/2000', api)
                tmp_df = pd.concat([tmp_df2, tmp_df], ignore_index=False)
            
            elif start < pd.to_datetime('01/01/2000') and end < pd.to_datetime('01/01/2000'):
                raise ValueError('Alpha Vantage is the best, but returns only stocks after 2000')
            
        
        else:
            tmp_df = reader.DataReader(ticker, api, start, end)
            tmp_df.reset_index(inplace=True)
            tmp_df.set_index("Date", inplace=True)
        
        
        return tmp_df





    #   ----------------------------------------------------- 
    def download_history_data_helper(self, ticker, t_start, t_end, verbose):
    #   ----------------------------------------------------- 
        """
        helper function for download_history_data.
        Uses APIs to download data.
         * Alpha Vantage
         * Yahoo Finance
         * Quandl
        
        ------
        input: ticker whose data is to be downloaded, start and end date 
               of interest, boolean for printing info
        output: dataframe with the desired output

        """
    #   ----------------------------------------------------- 

        if self.reload_sp500:
            # get from api
            
            try: 
                #default api: quandl
                api = 'alphavantage'
                t_df = self.retrieve_ticker_history(ticker, t_start, t_end, api)
                
            except Exception as err:
                try:
                    #second attempt with yahoo finance
                    api = 'yahoo'
                    t_df = self.retrieve_ticker_history(ticker, t_start, t_end, api)
                
                except Exception as err:
                    try:
                        #third attempt with AlphaVintage
                        api='quandl'
                        t_df = self.retrieve_ticker_history(ticker, t_start, t_end, api)
                    except Exception as err:
                        # if everything fails
                        if verbose:
                            print("No information for ticker '%s'"%ticker)
                        return pd.DataFrame()
            
            
            if self.overwrite is False and os.path.exists(self.data_folder+'{}.csv'.format(ticker)):
                if verbose:
                    print('file {}.csv already present'.format(ticker))
            else:
                t_df.to_csv(self.data_folder+'{}.csv'.format(ticker))
                if verbose:
                    print(" -> writing to file")
            
        return t_df





    #   ----------------------------------------------------- 
    def download_history_data(self, verbose=False):
    #   ----------------------------------------------------- 
        """
        Retrieves history of all the selected tickers:
         * using and API (quandl or yahoo finance)
         * from csv
        
        ------
        input: verbose - boolean.
        output: 2 dataframes, one for open and one for close values.
                columns: tickers
                index: date
        """
    #   ----------------------------------------------------- 

        if self.reload_sp500:
            self.retrieve_sp500_history()

            # close and open positions and date column
            data = reader.DataReader("MMM", 'yahoo', end=self.end, start=self.start)#, api_key=quandl_key)
            sp500_open = pd.DataFrame(index=data.index)
            sp500_close = pd.DataFrame(index=data.index)

            #loop over the current ticker
            for t, histoty in self.dict_history.items():
                
                t_open = []
                t_close = []
                
                # loop over the predecessor
                for it, record in enumerate(zip(histoty[0], histoty[1])):
                    ticker = record[0]
                    start_date = record[1]
                    
                    if verbose:
                        print("loading "+ticker)
                    
                    if it==0:
                        #to retrieve current ticker
                        end_date = self.end
                    
                    #retrieve history
                    ticker_df = self.download_history_data_helper(ticker, start_date, end_date, verbose)
                    ticker_df.columns = ticker_df.columns.map(lambda x: str(x) + '_'+t)
                    
                    if len(ticker_df)>0:
                        if 'Open_'+t in sp500_open:
                            mask = (sp500_open.index>start_date) & (sp500_open.index<=end_date)
                            sp500_open.loc[mask,'Open_'+t] = ticker_df['Open_'+t]
                            sp500_close.loc[mask,'Close_'+t] = ticker_df['Close_'+t]
                        else:
                            sp500_open = sp500_open.join(ticker_df['Open_'+t], how='outer')
                            sp500_close = sp500_close.join(ticker_df['Close_'+t], how='outer')
                    # the current start_date is the end_date of the previous element - one business day (extremes are also downloaded)
                    end_date = start_date - pd.Timedelta(1,'D')

            #write output dataframes
            if overwrite:
                sp500_open.to_csv(self.data_folder+'tmp_open.csv')
                sp500_close.to_csv(self.data_folder+'tmp_close.csv')

        else:
            sp500_close = pd.read_csv(self.data_folder+'sp500_open.csv')
            sp500_open = pd.read_csv(self.data_folder+'sp500_close.csv')
            sp500_open.rename(columns={ sp500_open.columns[0]: "Date" }, inplace = True)
            sp500_close.rename(columns={ sp500_close.columns[0]: "Date" }, inplace = True)
            sp500_open.set_index(pd.to_datetime(sp500_open.Date, format='%Y-%m-%d'), inplace = True)
            sp500_close.set_index(pd.to_datetime(sp500_close.Date, format='%Y-%m-%d'), inplace = True)
            mask = (sp500_open.index>self.start) & (sp500_open.index<=self.end)
            sp500_open = sp500_open[mask]
            sp500_close = sp500_close[mask]

        return sp500_open, sp500_close


