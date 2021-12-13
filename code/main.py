from data_loader import DataLoader

loader = DataLoader(reload_sp500=False, start='01/01/2000', end='01/01/2020', overwrite=False)
sp500_open, sp500_close = loader.download_history_data(verbose=False)
print(sp500_open)

