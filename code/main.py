from model_runner import Runner
import pandas as pd
import os

start_date = pd.to_datetime('01/01/2000')
end_date = pd.to_datetime('01/01/2020')

runner = Runner(verbose=True)
runner.prepare(start_date, end_date)#, number_of_stocks=100)


test_year = 2003
# model_type in ('lstm+soft_attention', 'lstm+custom_soft_attention', 'lstm+custom_general_attention' or 'lstm')
model_type = 'lstm+custom_soft_attention'
if not os.path.exists('model_'+model_type):
    os.makedirs('model_'+model_type)
runner.run(test_year, model_type=model_type, small_batch_size=False, feature_creation_space_size=120)


# ALTERNATIVELY: test many years. BUT Requires huge amount of time
#for test_year in range(2003,2020):
#    runner.run()

