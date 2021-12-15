from data_loader import DataLoader
from Statistics import Statistics
from data_preprocessor import Preprocessor
from model_trainer import Trainer
import numpy as np
import pandas as pd
import time


class Runner:
    """
    Class for running the model. Uses:
     
     - data_loader
     - data_preprocessor
     - model_trainer
     - Statistics
     
    Members:

     - verbose : boolean

    Methods:

     - prepare : handle data loading and preprocessing
     - run : run the training and performs prediction on test set
     - estimate_daily_return : estimates daily return trading the best 
           and worse predicted stocks
    """
    
    def __init__(self, verbose=False):

        self.verbose = verbose

    

    #   ----------------------------------------------------- 
    def prepare(self, start_date, end_date, reload_sp500=False, overwrite_sp500_csv=False, number_of_stocks=None):
    #   ----------------------------------------------------- 
    
        """
        Handle data loading.
        Results into two new class insatnces, self.sp500_close and self.sp500_open,
        dataframe storing historical close and open positions of sp500:
        columns: tickers, rows: business days
        Uses:
         - data_loader

        ------
        input  : - reload_sp500 [boolean] from internet, 
                 - overwrite_sp500_csv [boolean], 
                 - start_date and end_date (to download the data),
                 - number_of_stocks [int] to consider for the whole process (None=all)
        output : None

        """
    #   ----------------------------------------------------- 
        
        if self.verbose:
            print('loading data')
        loader = DataLoader(reload_sp500=reload_sp500, start=start_date, end=end_date, overwrite=overwrite_sp500_csv)
        self.sp500_open, self.sp500_close = loader.download_history_data(verbose=self.verbose)
        # default: use only columns with no nan (~330 columns)
        self.sp500_close = self.sp500_close.dropna(axis='columns')
        self.sp500_open = self.sp500_open.dropna(axis='columns')
        #unify column names
        self.sp500_open.columns = self.sp500_open.columns.str.replace('Open_','')
        self.sp500_close.columns = self.sp500_close.columns.str.replace('Close_','')
        
        #select only subset of dataset
        if number_of_stocks is not None:
            self.sp500_open = self.sp500_open.iloc[:,:number_of_stocks]
            self.sp500_close = self.sp500_close.iloc[:,:number_of_stocks]
 

    
    
    #   ----------------------------------------------------- 
    def estimate_daily_return(self, test_data, predictions):
    #   ----------------------------------------------------- 
        """
        Estimate daily return considering 10 stocks on which to
        go long, and 10 on which to go short

        """
    #   ----------------------------------------------------- 
        
        rets = pd.DataFrame([],columns=['Long','Short'])
        stocks_to_handle = 10
        
        for day in sorted(predictions.keys()):
            preds = predictions[day]
            test_returns = test_data[test_data[:,0]==day][:,-2]
            top_preds = predictions[day].argsort()[-stocks_to_handle:][::-1] 
            trans_long = test_returns[top_preds]
            worst_preds = predictions[day].argsort()[:stocks_to_handle][::-1] 
            trans_short = -test_returns[worst_preds]
            rets.loc[day] = [np.mean(trans_long),np.mean(trans_short)] 
        print('Result : ', rets.mean())  
        
        return rets     
    
    


    #   ----------------------------------------------------- 
    def run(self, test_year, model_type='lstm', small_batch_size=False, feature_creation_space_size=240):
    #   ----------------------------------------------------- 
    
        """
        Handle data preprocessing (mainly feature creation), model creation
        and running.
        Writes results on files in 'model_' + model_type folder
        Uses:
         - data_preprocessor
         - model_trainer
         - Statistics
        
        ------
        input  : 
                 - test_year of the model [int], 
                 - model_type ('lstm+soft_attention', 'lstm+custom_soft_attention' or 'lstm'),
                 - small_batch_size [bool] 
                   if True batch size is set to 32, if false to 512
                 - feature_creation_space_size [int] default 240, 
                   for custom attention use 120

        output : None

        """
    #   ----------------------------------------------------- 
       
        if model_type=='lstm+custom_soft_attention' and feature_creation_space_size==240:
            print('Attention: optimal number features for custom soft attention is 120')

        # set for the year under analysis
        df_open = self.sp500_open[self.sp500_open.index.year.isin([test_year-i for i in range(4)])]
        df_close = self.sp500_close[self.sp500_close.index.year.isin([test_year-i for i in range(4)])]
        df_open.reset_index(drop=True, inplace=True)
        df_close.reset_index(drop=True, inplace=True)
        
        #create labels
        preprocessor = Preprocessor(df_open, df_close)
        label = preprocessor.create_label()
        ticker_names = df_open.columns[1:].values
 
        # create features
        start = time.time()
        train_data = []
        test_data = []
        for ticker in ticker_names:
            ticker_train_data, ticker_test_data = preprocessor.create_features_for_single_ticker(ticker, test_year, feature_creation_space_size)
            train_data.append(ticker_train_data)
            test_data.append(ticker_test_data)
            if self.verbose:
                print('Creating features for:', ticker)

        # adjust and normalize dataset
        train_data = np.concatenate([x for x in train_data])
        test_data = np.concatenate([x for x in test_data])
        preprocessor.normalize_data(train_data, test_data)
        
        # train & predict
        trainer = Trainer(test_year, feature_creation_space_size, model_type=model_type, small_batch_size=small_batch_size)
        model, predictions = trainer.train(train_data, test_data)
        
        # results
        returns = self.estimate_daily_return(test_data, predictions)
        returns.to_csv('model_'+model_type+'/avg_daily_rets-'+str(test_year)+'.csv')
        result = Statistics(returns.sum(axis=1))
        print('\nAverage returns prior to transaction charges')
        result.shortreport() 
        
        #write to file
        with open("model_"+model_type+"/avg_returns.txt", "a") as myfile:
            res = '-'*30 + '\n'
            res += str(test_year) + '\n'
            res += 'Mean = ' + str(result.mean()) + '\n'
            res += 'Sharpe = '+str(result.sharpe()) + '\n'
            res += '-'*30 + '\n'
            myfile.write(res)


