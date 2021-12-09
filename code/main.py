from data_loader import DataLoader

#loader = DataLoader(reload_sp500=True, number_of_data=10, overwrite=False)
loader = DataLoader(reload_sp500=False, overwrite=False)
#df_open, df_close = loader.get_data(verbose=True)
loader.download_history_data(verbose=False)

#print(df_open.head())
#tmp = loader.get_sp500_changes()
#print(tmp)

# def create_stock_data(df_open,df_close,st,m=240):
#     st_data = pd.DataFrame([])
#     st_data['Date'] = list(df_close['Date'])
#     st_data['Name'] = [st]*len(st_data)
#     daily_change = df_close[st]/df_open[st]-1
#     for k in range(m)[::-1]:
#         st_data['IntraR'+str(k)] = daily_change.shift(k)
# 
#     nextday_ret = (np.array(df_open[st][1:])/np.array(df_close[st][:-1])-1)
#     nextday_ret = pd.Series(list(nextday_ret)+[np.nan])     
#     for k in range(m)[::-1]:
#         st_data['NextR'+str(k)] = nextday_ret.shift(k)
# 
#     close_change = df_close[st].pct_change()
#     for k in range(m)[::-1]:
#         st_data['CloseR'+str(k)] = close_change.shift(k)
# 
#     st_data['IntraR-future'] = daily_change.shift(-1)    
#     st_data['label'] = list(label[st])+[np.nan] 
#     st_data['Month'] = list(df_close['Date'].str[:-3])
#     st_data = st_data.dropna()
#     
#     trade_year = st_data['Month'].str[:4]
#     st_data = st_data.drop(columns=['Month'])
#     st_train_data = st_data[trade_year<str(test_year)]
#     st_test_data = st_data[trade_year==str(test_year)]
#     return np.array(st_train_data),np.array(st_test_data) 
