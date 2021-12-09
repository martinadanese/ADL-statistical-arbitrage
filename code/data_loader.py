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
        self.tickers = self.get_sp500_tickers()
        
        # remove double classe of the companies entering twice in sp500 (were not considered in the original paper):
        for t in ('GOOGL','DISCK', 'NWS', 'FOX', 'CMCSK', 'UA'):
            if t in self.tickers : self.tickers.remove(t)
        #dataframe with all past and present stock with added and removed date
        self.tickers_history = pd.DataFrame.from_dict({t : [self.start, self.end, None] for t in self.tickers},
                                 orient='index', columns=['Added_date', 'Removed_date', 'Predecessor'])
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
        
        missing_added = df_changes[df_changes.Added==0].copy(deep=True)
        #missing_removed = df_changes[df_changes.Removed==0].fillna(0).copy(deep=True)
        #missing_removed['Removed'] = missing_added['Removed'].values
        df_changes.loc[(df_changes.Removed==0), 'Removed'] = missing_added['Removed'].values
        df_changes = df_changes.loc[(df_changes.Added != 0)]
        
        df_changes.to_csv("sp500_changes.csv")
        return df_changes
    

    def get_tickers_history(self):
        return self.tickers_history

    def retrieve_sp500_history(self):
        df_changes = self.get_sp500_changes()
        #self.tickers_history contains only current values for the moment #.index.isin(df_changes.Added.values)
        df_tickers_rename = pd.read_csv('tickers_rename.csv', index_col=None, header=None, names=['Old_name', 'New_name', 'Date'])
        df_tickers_rename.Date = pd.to_datetime(df_tickers_rename.Date)
        
        # update added date (set union to have unique values)
        #repeated_t = []
        self.double = []
        for t in list(set(df_changes.Added.values) | set(self.tickers_history.index.values) | set(df_changes.Removed.values)):
            
            # !!!!!!!!!
            #if len(df_changes[df_changes.Added==t])+len(df_changes[df_changes.Removed==t])>2:
            #    repeated_t.append(t)
            #    print(t)
            #    continue
             
            # recorder added and removed tickers are the same (e.g. company merging)
            if t in df_changes[df_changes.Added==t].Removed.values:
                #if not anymore in sp500
                if (t not in self.tickers_history.index): # if not currently present need to add it
                    self.tickers_history.loc[t] = {'Added_date': df_changes[df_changes.Added==t].Date.values[0],
                                                   'Removed_date': df_changes[df_changes.Removed==t].Date.values[1], 
                                                   'Predecessor' : None}
                else:
                    continue
            
            elif any(df_changes.Added==t) and (not any(df_changes.Removed==t)) and (t not in self.tickers_history.index): 
                # only in df_changes.Added -> some merging or name change was not included in wikipedia database -> correct with df_tickers_rename
                real_t = df_tickers_rename.loc[df_tickers_rename.Old_name==t, 'New_name'].values[0]
                if real_t in self.tickers_history.index.values: 
                    # if real_t is still in sp500
                    self.tickers_history.loc[real_t, 'Added_date'] = df_changes[df_changes.Added==t].Date.values[0]
                    self.tickers_history.loc[real_t, 'Predecessor'] = df_changes[df_changes.Added==t].Removed.values[0]
                else: 
                    #if real_t is not in sp500 now
                    self.tickers_history.loc[real_t] = {'Added_date': df_changes[df_changes.Added==t].Date.values[0], 
                                          'Removed_date': df_changes[df_changes.Removed==real_t].Date.values[0], 
                                          'Predecessor' : df_changes[df_changes.Added==t].Removed.values[0]}

            elif (not any(df_changes.Added==t)) and any(df_changes.Removed==t) and (t not in self.tickers_history.index.values): 
                # only in df_changes.Removed -> is not currently in sp500 but was from start -> new row
                self.tickers_history.loc[t] = {'Added_date' : self.start, 
                                               'Removed_date' : df_changes[df_changes.Removed==t].Date.values[0],  
                                               'Predecessor' : None}
            
            elif (not any(df_changes.Added==t)) and (not any(df_changes.Removed==t)) and (t in self.tickers_history.index.values):
                # only in self.tickers_history
                continue
            
            #if removed and reput
            elif any(df_changes.Added==t) and any(df_changes.Removed==t) and (t in self.tickers):
                #if (len(df_changes[df_changes.Added==t])==1) and (len(df_changes[df_changes.Removed==t])==1): #always
                #still save the dates but check when downloading data
                self.tickers_history.loc[t] = {'Added_date' : df_changes[df_changes.Added==t].Date.values[0], 
                                               'Removed_date' : df_changes[df_changes.Removed==t].Date.values[0],  
                                               'Predecessor' : df_changes[df_changes.Added==t].Removed.values[0]}
                self.double.append(t)

            elif any(df_changes.Added==t) and any(df_changes.Removed==t) and (t not in self.tickers_history.index.values): 
                # in df_changes Added and Removed -> is not currently in sp500 but was -> new row
                self.tickers_history.loc[t] = {'Added_date': df_changes[df_changes.Added==t].Date.values[0],
                                               'Removed_date': df_changes[df_changes.Removed==t].Date.values[0], 
                                               'Predecessor' : df_changes[df_changes.Added==t].Removed.values[0]}
            
            #!!!!!!!!! ERROREEE
            elif (any(df_changes.Added==t)) and (not any(df_changes.Removed==t)) and (t in self.tickers_history.index.values):
                # in df_changes Added and self.tickers_history -> currently in sp500, but insert recently
                self.tickers_history.loc[t, 'Added_date'] = df_changes[df_changes.Added==t].Date.values[0]
                self.tickers_history.loc[t, 'Predecessor'] = df_changes[df_changes.Added==t].Removed.values[0]
                #self.tickers_history.loc[df_changes[df_changes.Added==t].Removed.values[0], '']
            
          
        self.tickers_history.to_csv('tickers_history.csv')
        #self.tickers_history.loc[self.tickers_history['Predecessor'].isin(repeated_t), 'Predecessor'] = None

  

        # OLD CHECKS

        #elif (not any(df_changes.Added==t)) and any(df_changes.Removed==t) and (t in self.tickers_history.index): 
        #shouldn't be a problem
        #    # in df_changes.Removed and self.tickers_history  
        #    print("Inconsistency type 2 found, ignored for now")
            #continue

        #elif (not any(df_changes.Added==t)) and (not any(df_changes.Removed==t)) and (t not in self.tickers_history.index): 
        #    # this should be impossible
        #    print("Inconsistency type 3 found for ", t)


    def get_sp500_history(self): #NOT USED
        """
        get history of sp500 (added and removed elements)
        
        ------
        input: None
        output: list of changes over time in sp500
        """
        data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        
        # table of current sp500 and set header column
        sp500 = data[0].iloc[:, [0,1,6,7]]
        sp500.columns = ['ticker', 'name', 'date' , 'cik']
        
        # Get rows where date is missing or not formatted correctly.
        mask = sp500['date'].str.strip().str.fullmatch('\d{4}-\d{2}-\d{2}')
        mask.loc[mask.isnull()] = False
        mask = mask == False
        
        # zerofill the cik code
        current = sp500.copy()
        current.loc[mask, 'date'] = '1900-01-01'
        current.loc[:, 'date'] = pd.to_datetime(current['date'])
        current.loc[:, 'cik'] = current['cik'].apply(str).str.zfill(10)
        
        
        # create a dataframe for additions and removals, and concatenate
        # Get the adjustments dataframe and rename columns
        adjustments = data[1]
        columns = ['date', 'ticker_added','name_added', 'ticker_removed', 'name_removed', 'reason']
        adjustments.columns = columns
        # Create additions dataframe.
        additions = adjustments[~adjustments['ticker_added'].isnull()][['date','ticker_added', 'name_added']]
        additions.columns = ['date','ticker','name']
        additions['action'] = 'added'
        # Create removals dataframe.
        removals = adjustments[~adjustments['ticker_removed'].isnull()][['date','ticker_removed','name_removed']]
        removals.columns = ['date','ticker','name']
        removals['action'] = 'removed'
        # Merge the additions and removals into one dataframe.
        historical = pd.concat([additions, removals])
        
        # add any tickers that are currently in the S&P 500 index but not in the Wikipedia history
        missing = current[~current['ticker'].isin(historical['ticker'])].copy()
        missing['action'] = 'added'
        missing = missing[['date','ticker','name','action', 'cik']]
        missing.loc[:, 'cik'] = current['cik'].apply(str).str.zfill(10)
        # merge         
        sp500_history = pd.concat([historical, missing])
        sp500_history = sp500_history.sort_values(by=['date','ticker'], ascending=[False, True])
        sp500_history = sp500_history.drop_duplicates(subset=['date','ticker'])
        #sp500_history.to_csv("sp500_history.csv")
        #historical.to_csv("sp500_changes.csv")
        return(historical)

        
    
    #def create_date_range(self):
    #    """
    #    """
    #    return rrule(DAILY, dtstart=self.start, until=self.end, byweekday=(MO,TU,WE,TH,FR))


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


    def get_data(self, verbose=False):
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
        




    def download_history_data_helper(self, ticker, tmp_open, tmp_close, t_start, t_end, dates, verbose):

        if self.reload_sp500:
            # get from api
            
            try: 
                #default api: quandl
                api = 'quandl'
                tmp_df = self.retrieve_tricker_history(ticker, t_start, t_end, api)
                
                #else:
                    
                
                if self.overwrite is False and os.path.exists(self.data_folder+'{}.csv'.format(ticker)):
                    print('file {}.csv already present'.format(ticker))
                else:
                    tmp_df.to_csv(self.data_folder+'{}.csv'.format(ticker))
                
                #make an unique list of time values
                tmp_open = tmp_open + tmp_df['Open'].to_list()
                tmp_close = tmp_close + tmp_df['Close'].to_list()
                if len(tmp_df) > len(dates):
                    dates = tmp_df.Date

            except Exception:
            #    try:
            #        #second attempt with yahoo finance
            #        api = 'yahoo'
            #        tmp_df = self.retrieve_history(ticker, api)
            #        
            #        if self.overwrite is False and os.path.exists(self.data_folder+'{}.csv'.format(ticker)):
            #            print('file {}.csv already present'.format(ticker))
            #        else:
            #            tmp_df.to_csv(self.data_folder+'{}.csv'.format(ticker))
            #        if len(dates) < len(tmp_df.index):
            #            dates = tmp_df.index
            #        tmp_open.append(tmp_df['Open'].to_list())
            #        tmp_close.append(tmp_df['Close'])
            #        found.append(ticker)
            #    except Exception:
            #        print("No information for ticker '%s'"%ticker)
            #        missed.append(ticker)
            #        continue
                print("No information for ticker '%s'"%ticker)
                #missed.append(ticker)
        else:
        #    #read from csv
            try:
                tmp_df = pd.read_csv(self.data_folder+'{}.csv'.format(ticker))
                tmp_df.index = pd.to_datetime(tmp_df['Date'])
                #mask = (tmp_df.index>self.start) & (tmp_df.index<=self.end)
                start = (t_start if t_start > self.start else self.start)
                end = (t_end if t_end < self.end else self.end)
                mask = (tmp_df.index>start) & (tmp_df.index<=end)
                #tmp_open.append(tmp_df.loc[mask]['Open'].to_list())
                #tmp_close.append(tmp_df.loc[mask]['Close'].to_list()) 
                tmp_open = tmp_open + tmp_df.loc[mask]['Open'].to_list()
                tmp_close = tmp_close + tmp_df.loc[mask]['Close'].to_list()
                if len(tmp_df) > len(dates):
                    dates = tmp_df.Date
            except Exception:
                print("No information for ticker '%s'" % ticker)
                #missed.append(ticker)
                #continue

        return tmp_open, tmp_close, dates





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

        sp500_open = []
        sp500_close = []
        dates = []
        #to track unretrievable data
        missed = []

        #loop over the current ticker
        for ticker in self.tickers:

            #store whole history of a single ticker its predecessors
            tmp_open = []
            tmp_close = []

            # loop over the predecessor
            while ticker is not None:
                
                if verbose:
                    print("loading "+ticker)
                

                if ticker not in self.double:#self.tickers_history['Added_date'][ticker] < self.tickers_history['Removed_date'][ticker]:
                    print(ticker)
                    t_start = self.tickers_history.loc[ticker, 'Added_date']
                    t_end = self.tickers_history.loc[ticker, 'Removed_date']
                    #case it was a double already seen, handle I part
                    if t_end < t_start:
                        tmp_open, tmp_close, dates = self.download_history_data_helper(ticker, tmp_open, tmp_close, self.start, t_end, dates, verbose)
                    else:
                        tmp_open, tmp_close, dates = self.download_history_data_helper(ticker, tmp_open, tmp_close, t_start, t_end, dates, verbose)
                else:
                    # if it is a double handle only II part here
                    t_start = self.tickers_history.loc[ticker, 'Added_date']
                    t_end = self.end
                    tmp_open, tmp_close, dates = self.download_history_data_helper(ticker, tmp_open, tmp_close, t_start, t_end, dates, verbose)
                    self.double.remove(ticker)
                #try:
                ticker = self.tickers_history['Predecessor'][ticker]
                #except Exception:
                #    break
            
            #out of while loop
            sp500_open.append(pd.Series(tmp_open))
            sp500_close.append(pd.Series(tmp_close))
        #write output dataframes
        df_open = pd.DataFrame(sp500_open)
        df_close = pd.DataFrame(sp500_close)
        df_open = df_open.transpose()
        df_close = df_close.transpose()
        df_open.columns = self.tickers
        df_close.columns = self.tickers

        df_open.set_index(dates, inplace = True)
        df_close.set_index(dates, inplace = True)
        
        df_open.to_csv('tmp_open.csv')
        df_close.to_csv('tmp_close.csv')
        
        return df_open, df_close
