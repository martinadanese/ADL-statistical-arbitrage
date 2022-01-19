from flask import flash, Flask, render_template, request, redirect, url_for, send_from_directory
from data_preprocessor import Preprocessor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from attention import Bahdanau
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import os
import sys


ALLOWED_EXTENSIONS = {'csv'}

# import the Flask object to create a Flask application instance
app = Flask(__name__)
# to configure 
app.config['UPLOAD_FOLDER'] = './static/' 
#app.config['MAX_CONTENT_PATH']


def estimate_daily_return(test_data, predictions):
    """
    Estimate daily return considering 10 stocks on which to
    go long, and 10 on which to go short

    """
    
    rets = pd.DataFrame([],columns=['Long','Short'])
    stocks_to_handle = 10
    first = True

    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:,0]==day][:,-2]
        top_preds = predictions[day].argsort()[-stocks_to_handle:][::-1] 
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:stocks_to_handle][::-1] 
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long),np.mean(trans_short)] 
        if first:
            res_long = top_preds
            res_short = worst_preds
            first = False
    print('Result : ', rets.mean())  
    
    return rets, res_long, res_short



def reshape(x):
    """
    swap axes
    """
    x = np.array(np.split(x, 3, axis=1))
    x = np.swapaxes(x, 0, 1)
    x = np.swapaxes(x, 1, 2)
    return x



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def upload_form():
    return render_template('index.html')

open_submitted = False
close_submitted = False
open_name = ''
close_name = ''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # empty file submitted 
        if file.filename == '':
            # run option returns the result page
            if request.form['submit_button'] == 'Run':
                line1, line2, line3 = compute()
                return render_template('results.html', line1=line1, line2=line2, line3=line3)
            # just empty file
            else:
                flash('No selected file')
                return redirect(request.url)
        #if there is a file and it is an allowed format
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # check uploaded files
            if request.form['submit_button'] == 'UploadClose':
                global close_submitted
                global close_name
                close_submitted = True
                close_name = app.config['UPLOAD_FOLDER']+'/'+filename
            elif request.form['submit_button'] == 'UploadOpen':
                global open_submitted
                global open_name
                open_submitted = True
                open_name = app.config['UPLOAD_FOLDER']+'/'+filename
            
            return render_template('index.html')#uploaded_file()

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def compute():
    """
    when the run button is pressed:
    run the prediction 
    """
    print('running')
    sp500_close = pd.read_csv(close_name)
    sp500_open = pd.read_csv(open_name)
    sp500_open.rename(columns={ sp500_open.columns[0]: "Date" }, inplace = True)
    sp500_close.rename(columns={ sp500_close.columns[0]: "Date" }, inplace = True)
    sp500_open.set_index(pd.to_datetime(sp500_open.Date, format='%Y-%m-%d'), inplace = True)
    sp500_close.set_index(pd.to_datetime(sp500_close.Date, format='%Y-%m-%d'), inplace = True)
    test_year = sp500_close.index.year.values[-1]
    # model_runner : prepare
    sp500_close = sp500_close.dropna(axis='columns')
    sp500_open = sp500_open.dropna(axis='columns')
    sp500_open.columns = sp500_open.columns.str.replace('Open_','')
    sp500_close.columns = sp500_close.columns.str.replace('Close_','')
    # model_runner : run 
    # Optimal parameters for Bahdanau
    feature_creation_space_size = 120
    #
    df_open = sp500_open[sp500_open.index.year.isin([test_year-i for i in range(4)])]
    df_close = sp500_close[sp500_close.index.year.isin([test_year-i for i in range(4)])]
    df_open.reset_index(drop=True, inplace=True)
    df_close.reset_index(drop=True, inplace=True)
    # create labels
    preprocessor = Preprocessor(df_open, df_close)
    label = preprocessor.create_label()
    ticker_names = df_open.columns[1:].values
    # create features
    test_data = []
    for ticker in ticker_names:
        _, ticker_test_data = preprocessor.create_features_for_single_ticker(ticker, test_year, feature_creation_space_size)
        test_data.append(ticker_test_data)
    # adjust and normalize dataset
    test_data = np.concatenate([x for x in test_data])
    scaler = RobustScaler()
    scaler.fit(test_data[:,2:-2])
    test_data[:,2:-2] = scaler.transform(test_data[:,2:-2]) 
    
    # model_trainer : train (not actually training here)
    model = load_model(app.config['UPLOAD_FOLDER']+"/2003-E29.h5", custom_objects={"Bahdanau": Bahdanau}, compile=False)
    print('Model loaded. Start serving...')
    dates = list(set(test_data[:,0]))
    predictions = {}
    for day in dates:
        test_day = test_data[test_data[:,0]==day]
        test_day = reshape(test_day[:,2:-2])
        test_day = np.asarray(test_day).astype('float32')
        predictions[day] = model.predict(test_day)[:,1]
    returns, res_long, res_short = estimate_daily_return(test_data, predictions)
    
    col_long = []
    col_short = []
    for long_pos, short_pos in zip(res_long, res_short):
        col_long.append(sp500_open.columns[long_pos])
        col_short.append(sp500_open.columns[short_pos])

    line1 = 'The average daily return on this portfolio is {:.2f}%'.format(100*np.mean(returns.sum(axis=1)))
    line2 = 'As first action you should long: {}'.format(col_long)
    line3 = 'and short: {}'.format(col_short)
    return line1, line2, line3



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


