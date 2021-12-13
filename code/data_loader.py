import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import pandas as pd
#!pip install pandas-datareader --upgrade
import pandas_datareader.data as reader
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
        
        #else:
        #    self.data_folder = '../data/15years_data/'
        #    self.tickers = [i[0:-4] for i in os.listdir(self.data_folder)]
        
        #load only selected data
        #if number_of_data is not None:
        #    self.tickers = self.tickers[0:number_of_data]
         
        
    def get_sp500_tickers(self):
        """
        get history of the current sp500 stocks
        
        ------
        input: None
        output: list of tickers in sp500
        """
        
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol']
    
        def to_datetime(date):
            return dt.datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d')

    
    def get_sp500_changes(self):
        df_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df_changes = pd.DataFrame({'Date' : df_list[1].Date.Date, 'Added' : df_list[1].Added.Ticker, 'Removed' : df_list[1].Removed.Ticker})
        df_changes.Date = pd.to_datetime(df_changes.Date, format='%B %d, %Y')
        
        #fori i in values
        df_changes.fillna(0, inplace=True)
        
        # remove double classe of the companies entering twice in sp500 (were not considered in the original paper):
        for t in ('GOOGL','DISCK', 'NWS', 'FOX', 'CMCSK', 'UA'):
            df_changes.drop(index=df_changes[(df_changes.Added==t) | (df_changes.Removed==t)].index, inplace=True)
        
        #handle cases in which remove & additions were done in two separate days
        missing_added = df_changes[df_changes.Added==0].copy(deep=True)
        df_changes.loc[(df_changes.Removed==0), 'Removed'] = missing_added['Removed'].values
        df_changes = df_changes.loc[(df_changes.Added != 0)]
        
        #save to file
        df_changes.to_csv("sp500_changes.csv")
        return df_changes
    

    def get_tickers_history(self):
        return self.tickers_history


    def Preceding(self, t, df_changes):
        return preceding, date, df_changes


    def retrieve_sp500_history(self):
        df_changes = self.get_sp500_changes()
        #self.tickers_history contains only current values for the moment #.index.isin(df_changes.Added.values)
        df_tickers_rename = pd.read_csv('tickers_rename.csv', index_col=None, header=None, names=['Old_name', 'New_name', 'Date'])
        df_tickers_rename.Date = pd.to_datetime(df_tickers_rename.Date)
        
        # update added date (set union to have unique values)
        # dictionary s.t. Ticker_i = [[Preceiding to Ticker_i, Preceiding to the Preceiding to Ticker_i, ....], [Date of change with preceding, date of change with preceding of the preceding...]]
        self.dict_history = {t : [[t],[]] for t in self.tickers}
        for t in self.tickers:
            
            ticker = t
            previous_date = self.end
            while any(df_changes.Added==ticker):
                # take predecessor and date of ticker addition. If more than one, take most recent
                predecessor = df_changes.loc[(df_changes.Date == df_changes.loc[df_changes.Added==ticker, 'Date'].max()) & (df_changes.Added==ticker), 'Removed'].values[0]
                date = df_changes.loc[(df_changes.Date == df_changes.loc[df_changes.Added==ticker, 'Date'].max()) & (df_changes.Added==ticker), 'Date'].values[0]
                #remove column from df_changes so that the max date can always be taken
                df_changes = df_changes.drop(df_changes[(df_changes.Date == df_changes.loc[df_changes.Added==ticker, 'Date'].max()) & (df_changes.Added==ticker)].index)
                if pd.to_datetime(date) > pd.to_datetime(previous_date):
                    continue
                self.dict_history[t][0].append(predecessor)
                self.dict_history[t][1].append(date)
                ticker = predecessor
                previous_date = date

            self.dict_history[t][1].append(self.start)






















        #    # recorder added and removed tickers are the same (e.g. company merging)
        #    if t in df_changes[df_changes.Added==t].Removed.values:
        #        #if not anymore in sp500
        #        if (t not in self.tickers_history.index): # if not currently present need to add it
        #            self.tickers_history.loc[t] = {'Added_date': df_changes[df_changes.Added==t].Date.values[0],
        #                                           'Removed_date': df_changes[df_changes.Removed==t].Date.values[1], 
        #                                           'Predecessor' : None}
        #        else:
        #            continue
        #    
        #    #ALL PRESENT
        #    if t in self.tickers_history.index:

        #        # /////]--[///]--[///>   =  2 added and 2 removed
        #        if (len(df_changes[df_changes.Added==t])==2) and (len(df_changes[df_changes.Removed==t])==2):
        #            












        #  
        #    elif any(df_changes.Added==t) and (not any(df_changes.Removed==t)) and (t not in self.tickers_history.index): 
        #        # only in df_changes.Added -> some merging or name change was not included in wikipedia database -> correct with df_tickers_rename
        #        real_t = df_tickers_rename.loc[df_tickers_rename.Old_name==t, 'New_name'].values[0]
        #        if real_t in self.tickers_history.index.values: 
        #            # if real_t is still in sp500
        #            self.tickers_history.loc[real_t, 'Added_date'] = df_changes[df_changes.Added==t].Date.values[0]
        #            self.tickers_history.loc[real_t, 'Predecessor'] = df_changes[df_changes.Added==t].Removed.values[0]
        #        else: 
        #            #if real_t is not in sp500 now
        #            self.tickers_history.loc[real_t] = {'Added_date': df_changes[df_changes.Added==t].Date.values[0], 
        #                                  'Removed_date': df_changes[df_changes.Removed==real_t].Date.values[0], 
        #                                  'Predecessor' : df_changes[df_changes.Added==t].Removed.values[0]}

        #    elif (not any(df_changes.Added==t)) and any(df_changes.Removed==t) and (t not in self.tickers_history.index.values):  # /////|------>
        #        # only in df_changes.Removed -> is not currently in sp500 but was from start -> new row
        #        self.tickers_history.loc[t] = {'Added_date' : self.start, 
        #                                       'Removed_date' : df_changes[df_changes.Removed==t].Date.values[0],  
        #                                       'Predecessor' : None}
        #    
        #    elif (not any(df_changes.Added==t)) and (not any(df_changes.Removed==t)) and (t in self.tickers_history.index.values):   
        #        # only in self.tickers_history
        #        continue
        #    
        #    #if removed and reput
        #    elif any(df_changes.Added==t) and any(df_changes.Removed==t) and (t in self.tickers):
        #        #if (len(df_changes[df_changes.Added==t])==1) and (len(df_changes[df_changes.Removed==t])==1): #always
        #        #still save the dates but check when downloading data
        #        self.tickers_history.loc[t] = {'Added_date' : df_changes[df_changes.Added==t].Date.values[0], 
        #                                       'Removed_date' : df_changes[df_changes.Removed==t].Date.values[0],  
        #                                       'Predecessor' : df_changes[df_changes.Added==t].Removed.values[0]}
        #        self.double.append(t)

        #    elif any(df_changes.Added==t) and any(df_changes.Removed==t) and (t not in self.tickers_history.index.values): 
        #        # in df_changes Added and Removed -> is not currently in sp500 but was -> new row
        #        self.tickers_history.loc[t] = {'Added_date': df_changes[df_changes.Added==t].Date.values[0],
        #                                       'Removed_date': df_changes[df_changes.Removed==t].Date.values[0], 
        #                                       'Predecessor' : df_changes[df_changes.Added==t].Removed.values[0]}
        #    
        #    #!!!!!!!!! ERROREEE
        #    elif (any(df_changes.Added==t)) and (not any(df_changes.Removed==t)) and (t in self.tickers_history.index.values):
        #        # in df_changes Added and self.tickers_history -> currently in sp500, but insert recently
        #        # NON DIPENDE DAL TICKERS MESSO DENTRO DUE VOLTE PERCHÉ QUESTO DÀ COMUNQUE ERRORE
        #        #added: expected:  ------[/////>
        #        # if len(df_changes[df_changes.Added==t])==1: # no  /////]----[////>
        #        # if len(df_changes[df_changes.Removed==t])==0: #no ---[////]----[////>
        #        #predecessor: expected   /////]----->
        #        if df_changes[df_changes.Added==t].Removed.values[0] not in self.tickers_history.index.values: # some are ////]-----[////> handled above
        #            # if len(df_changes[df_changes.Added==df_changes[df_changes.Added==t].Removed.values[0]])==1:
        #            # if len(df_changes[df_changes.Added==t].Removed.values) ==1:
        #            self.tickers_history.loc[t, 'Added_date'] = df_changes[df_changes.Added==t].Date.values[0]
        #            self.tickers_history.loc[t, 'Predecessor'] = df_changes[df_changes.Added==t].Removed.values[0]
        #            #print(t, "found predecessor: ", df_changes[df_changes.Added==t].Removed.values)
        #    
        #  
        #self.tickers_history.to_csv('tickers_history.csv')
        #self.tickers_history.loc[self.tickers_history['Predecessor'].isin(repeated_t), 'Predecessor'] = None



        


    def retrieve_tricker_history(self, ticker, start, end, api):
        """
        helper function that retrieves history of single stock
        
        ------
        input: ticker name, desired API
        output: dataframe with history
        """
        if api == 'quandl':
            tmp_df = reader.DataReader("WIKI/"+ticker, 'quandl', start, end, api_key=self.quandl_key)
        else:
            tmp_df = reader.DataReader(ticker, api, start, end)
        
        tmp_df.reset_index(inplace=True)
        tmp_df.set_index("Date", inplace=True)
        
        return tmp_df





    def download_history_data_helper(self, ticker, t_open, t_close, t_start, t_end, dates_column, verbose):

        if self.reload_sp500:
            # get from api
            
            try: 
                #default api: quandl
                api = 'quandl'
                tmp_df = self.retrieve_tricker_history(ticker, t_start, t_end, api)
                
            except Exception:
                try:
                    #second attempt with yahoo finance
                    api = 'yahoo'
                    tmp_df = self.retrieve_history(ticker, api)
                
                except Exception:
                    print("No information for ticker '%s'"%ticker)
                    return t_open, t_close, dates_column
            
            
            if self.overwrite is False and os.path.exists(self.data_folder+'{}.csv'.format(ticker)):
                if verbose:
                    print('file {}.csv already present'.format(ticker))
            else:
                tmp_df.to_csv(self.data_folder+'{}.csv'.format(ticker))
                if verbose:
                    print(" -> writing to file")
            
            #make an unique list of time values
            t_open = t_open + tmp_df['Open'].to_list()
            t_close = t_close + tmp_df['Close'].to_list()
            if len(tmp_df) > len(dates_column):
                dates_column = tmp_df.index.values

        else:
        #    #read from csv
            try:
                tmp_df = pd.read_csv(self.data_folder+'{}.csv'.format(ticker))
                if len(tmp_df) > len(dates_column):
                    dates_column = tmp_df.Date
                tmp_df.index = pd.to_datetime(tmp_df['Date'])
                #mask = (tmp_df.index>self.start) & (tmp_df.index<=self.end)
                #start = (t_start if t_start > self.start else self.start)
                #end = (t_end if t_end < self.end else self.end)
                mask = (tmp_df.index>t_start) & (tmp_df.index<=t_end)
                #tmp_open.append(tmp_df.loc[mask]['Open'].to_list())
                #tmp_close.append(tmp_df.loc[mask]['Close'].to_list()) 
                t_open = t_open + tmp_df.loc[mask]['Open'].to_list()
                t_close = t_close + tmp_df.loc[mask]['Close'].to_list()
            except Exception:
                print("No information for ticker '%s'" % ticker)
                #missed.append(ticker)
                #continue

        return t_open, t_close, dates_column





    def download_history_data(self, verbose=False):
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

        self.retrieve_sp500_history()

        # close and open positions and date column
        sp500_open = []
        sp500_close = []
        dates_column = []
        #to track unretrievable data
        missed = []

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
                t_open, t_close, dates_column = self.download_history_data_helper(ticker, t_open, t_close, start_date, end_date, dates_column, verbose)
                # the current start_date is the end_date of the previous element - one business day (extremes are also downloaded)
                bd = pd.tseries.offsets.BusinessDay(n=1)
                end_date = start_date - bd

            #out of while loop
            sp500_open.append(pd.Series(t_open))
            sp500_close.append(pd.Series(t_close))
        #write output dataframes
        df_open = pd.DataFrame(sp500_open)
        df_close = pd.DataFrame(sp500_close)
        df_open = df_open.transpose()
        df_close = df_close.transpose()
        df_open.columns = self.tickers
        df_close.columns = self.tickers
        df_open.to_csv('tmp_open.csv')
        df_close.to_csv('tmp_close.csv')

        df_open.set_index(pd.to_datetime(dates_column), inplace = True)
        df_close.set_index(pd.to_datetime(dates_column), inplace = True)
        
        df_open.to_csv('tmp_open.csv')
        df_close.to_csv('tmp_close.csv')
        
        return df_open, df_close
