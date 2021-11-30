from data_loader import DataLoader

loader = DataLoader(reload_sp500=False)
df_open, df_close = loader.get_data()
print(df_open.tail())
