from data_loader import DataLoader

loader = DataLoader(reload_sp500=False, number_of_data=10)
df_open, df_close = loader.get_data(verbose=True)

print(df_open.head())
